from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel, QLineEdit, QScrollArea
import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from openseespy.opensees import *

# ------------------------
# Structural Model Data
# ------------------------
E = 200e9       # N/m^2
A = 0.005       # m^2

nodes = np.array([[0,0],
                  [6,0],
                  [12,0],
                  [18,0],
                  [24,0],
                  [18,6],
                  [12,6],
                  [6,6]])

members = np.array([[1,2],
                    [2,3],
                    [3,4],
                    [4,5],
                    [5,6],
                    [6,7],
                    [7,8],
                    [1,8],
                    [2,8],
                    [3,7],
                    [4,6],
                    [2,7],
                    [4,7]])

load_nodes = [1, 2, 3, 4, 5]    # moving load path (node numbers)
N_MEMBERS = members.shape[0]
N_STEPS = len(load_nodes)

# ------------------------
# PyQt5 GUI
# ------------------------
class TrussApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Truss Analysis with OpenSeesPy — Animated Influence Lines")
        self.setGeometry(50, 50, 1400, 800)

        layout = QHBoxLayout(self)

        # Left panel (controls)
        control_layout = QVBoxLayout()
        self.load_label = QLabel("Load Value (N):")
        self.load_input = QLineEdit("-50000")
        self.run_button = QPushButton("Run Analysis")
        self.run_button.clicked.connect(self.run_analysis)

        control_layout.addWidget(self.load_label)
        control_layout.addWidget(self.load_input)
        control_layout.addWidget(self.run_button)
        control_layout.addStretch()
        layout.addLayout(control_layout, 1)

        # Right panel: two canvases vertically (deflection above, influence grid below)
        right_layout = QVBoxLayout()

        # Canvas: Deflected shape
        self.fig_def, self.ax_def = plt.subplots(figsize=(6,4))
        self.canvas_def = FigureCanvas(self.fig_def)
        right_layout.addWidget(self.canvas_def, 2)

        # Canvas: Influence lines grid inside a scroll area (in case screen small)
        self.fig_inf = plt.figure(figsize=(9, 8))
        self.canvas_inf = FigureCanvas(self.fig_inf)
        right_layout.addWidget(self.canvas_inf, 3)

        layout.addLayout(right_layout, 4)

        # Storage
        self.all_disps = []            # list of (n_nodes x 2) arrays
        self.member_forces = None      # (n_steps x n_members) array

        # Animation holders
        self.ani = None
        self.ani_inf = None

        # plotting objects placeholders
        self.deformed_lines = []
        self.load_marker = None
        self.inf_axes = []
        self.inf_lines = []
        self.inf_point_markers = []
        self.inf_vlines = []

    def run_analysis(self):
        # Clear previous
        self.ax_def.clear()
        self.fig_inf.clear()
        self.all_disps.clear()
        self.member_forces = None

        # Read load value
        try:
            load_value = float(self.load_input.text())
        except:
            load_value = -50000.0

        # Run OpenSees
        self.run_opensees(load_value)

        # Prepare & start animations
        self.setup_deflection_plot()
        self.setup_influence_grid()
        self.start_animations()

    def run_opensees(self, load_value):
        # Build and analyze model for each load position; store displacements and member axial forces
        wipe()
        model('basic', '-ndm', 2, '-ndf', 2)

        for i, n in enumerate(nodes):
            node(i+1, float(n[0]), float(n[1]))

        uniaxialMaterial("Elastic", 1, E)
        for i, mbr in enumerate(members):
            element("Truss", i+1, int(mbr[0]), int(mbr[1]), A, 1)

        # Supports
        fix(1,1,1)   # pin
        fix(5,0,1)   # roller

        system("BandSPD")
        numberer("RCM")
        constraints("Plain")
        integrator("LoadControl", 1.0)
        algorithm("Linear")
        analysis("Static")

        base_tag = 1000
        local_member_forces = []

        for step_idx, ntag in enumerate(load_nodes, start=1):
            ts_tag = base_tag + 2*step_idx - 1
            pat_tag = base_tag + 2*step_idx

            timeSeries("Constant", ts_tag)
            pattern("Plain", pat_tag, ts_tag)
            load(ntag, 0.0, load_value)

            ok = analyze(1)
            if ok != 0:
                print(f"Warning: analysis return code {ok} at step {step_idx}")

            # store nodal displacements
            disp = []
            for nid in range(1, len(nodes)+1):
                ux = nodeDisp(nid, 1)
                uy = nodeDisp(nid, 2)
                disp.append([ux, uy])
            self.all_disps.append(np.array(disp))

            # store axial forces for all members (in kN)
            mbrForces = []
            for i in range(1, N_MEMBERS+1):
                axial = basicForce(i)[0] / 1000.0
                mbrForces.append(axial)
            local_member_forces.append(mbrForces)

            # cleanup pattern & timeseries tags (not all OpenSees builds allow remove of timeseries)
            try:
                remove('loadPattern', pat_tag)
            except Exception:
                pass
            try:
                remove('timeSeries', ts_tag)
            except Exception:
                pass

        self.member_forces = np.array(local_member_forces)  # shape: (n_steps, n_members)

    def setup_deflection_plot(self):
        self.ax_def.set_aspect('equal', 'box')
        self.ax_def.set_title("Deflected shape under moving load")
        self.ax_def.set_xlabel("Distance (m)")
        self.ax_def.set_ylabel("Distance (m)")
        self.ax_def.grid(True)

        xFac = 500  # scaling for display
        self.xFac = xFac

        # plot undeformed (grey)
        for mbr in members:
            i, j = int(mbr[0]) - 1, int(mbr[1]) - 1
            self.ax_def.plot([nodes[i,0], nodes[j,0]], [nodes[i,1], nodes[j,1]], color='0.5', lw=0.75)

        # prepare deformed member Line2D objects
        self.deformed_lines = []
        for _ in members:
            line, = self.ax_def.plot([], [], 'r-', lw=1.5)
            self.deformed_lines.append(line)

        # prepare a marker indicating the moving load (a larger filled circle)
        self.load_marker, = self.ax_def.plot([], [], 'o', ms=10, markerfacecolor='blue', markeredgecolor='k')

        self.ax_def.set_xlim(np.min(nodes[:,0]) - 2, np.max(nodes[:,0]) + 2)
        self.ax_def.set_ylim(np.min(nodes[:,1]) - 2, np.max(nodes[:,1]) + 2)

    def setup_influence_grid(self):
        # Create a grid of subplots (13). We'll do 4x4 to fit 13 (one empty).
        ncols = 4
        nrows = 4
        gs = self.fig_inf.add_gridspec(nrows, ncols, hspace=0.6, wspace=0.45)

        self.inf_axes = []
        self.inf_lines = []
        self.inf_point_markers = []
        self.inf_vlines = []

        x_positions = load_nodes  # x-axis will show node numbers where load is placed

        for m in range(N_MEMBERS):
            r = m // ncols
            c = m % ncols
            ax = self.fig_inf.add_subplot(gs[r, c])
            ax.set_title(f"Member {m+1}: {members[m,0]}-{members[m,1]}", fontsize=9)
            ax.set_xlabel("Load at node")
            ax.set_ylabel("Axial (kN)")
            ax.set_xticks(x_positions)
            ax.grid(True)

            # initial empty line and marker
            line, = ax.plot([], [], marker='o', lw=1)
            # marker for current point
            point, = ax.plot([], [], 's', ms=6, markerfacecolor='red', markeredgecolor='k')
            # vertical line to show where load currently is
            vline = ax.axvline(x_positions[0], color='gray', linestyle='--', lw=0.8)

            self.inf_axes.append(ax)
            self.inf_lines.append(line)
            self.inf_point_markers.append(point)
            self.inf_vlines.append(vline)

        # If grid has extra subplot(s) (like the 16th), clear it
        for m in range(N_MEMBERS, ncols*nrows):
            r = m // ncols
            c = m % ncols
            self.fig_inf.add_subplot(gs[r, c]).axis('off')

        # tighten layout
        self.fig_inf.subplots_adjust(hspace=0.6, wspace=0.45)

    def start_animations(self):
        # single FuncAnimation that updates both deflection and all influence subplots
        n_frames = len(self.all_disps)
        interval = 900  # ms per frame (tweak as needed)

        def update(frame):
            # ---- Deflection update ----
            disp = self.all_disps[frame]
            for idx, mbr in enumerate(members):
                i, j = int(mbr[0]) - 1, int(mbr[1]) - 1
                xi, yi = nodes[i]
                xj, yj = nodes[j]
                uxi, uyi = disp[i]
                uxj, uyj = disp[j]
                xs = [xi + uxi*self.xFac, xj + uxj*self.xFac]
                ys = [yi + uyi*self.xFac, yj + uyj*self.xFac]
                self.deformed_lines[idx].set_data(xs, ys)

            # update load marker position on deflection plot (show at loaded node + displacement)
            load_node_num = load_nodes[frame]
            load_idx = int(load_node_num) - 1
            ux, uy = disp[load_idx]
            lx = nodes[load_idx, 0] + ux*self.xFac
            ly = nodes[load_idx, 1] + uy*self.xFac
            self.load_marker.set_data([lx], [ly])

            # ---- Influence lines update (for each member) ----
            x = np.array(load_nodes)
            for m in range(N_MEMBERS):
                y_all = self.member_forces[:, m]  # length = n_steps
                # plot cumulative points upto current frame
                xs = x[:frame+1]
                ys = y_all[:frame+1]
                self.inf_lines[m].set_data(xs, ys)

                # set current highlighted marker at (current load position, value)
                current_x = x[frame]
                current_y = y_all[frame]
                self.inf_point_markers[m].set_data([current_x], [current_y])

                # move vline to current load position
                self.inf_vlines[m].set_xdata([current_x, current_x])

                # adjust y-limits slightly to ensure visibility
                ax = self.inf_axes[m]
                ymin, ymax = np.min(y_all) , np.max(y_all)
                # add small margin
                margin = max(1e-3, 0.08*(ymax - ymin) if ymax != ymin else 1.0)
                ax.set_ylim(ymin - margin, ymax + margin)

            # draw canvases
            self.canvas_def.draw_idle()
            self.canvas_inf.draw_idle()

            return []  # not using blit to avoid complexity

        # stop previous animations if any
        if self.ani is not None:
            self.ani.event_source.stop()
        if self.ani_inf is not None:
            self.ani_inf.event_source.stop()

        # create one animation that runs update for all plots simultaneously
        self.ani = animation.FuncAnimation(self.fig_def, update,
                                           frames=n_frames,
                                           interval=interval,
                                           blit=False,
                                           repeat=True)
        # we also need to tie the same update to the influence figure's animation loop so both are redrawn
        # create a second animation (same update) that uses the influence figure as the target
        self.ani_inf = animation.FuncAnimation(self.fig_inf, update,
                                               frames=n_frames,
                                               interval=interval,
                                               blit=False,
                                               repeat=True)

        # initial draw
        self.canvas_def.draw()
        self.canvas_inf.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrussApp()
    window.show()
    sys.exit(app.exec_())
