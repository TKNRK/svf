import os
import logging
import sys
from sn.qt import *
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.Qt import QFont
from constants import *
from svf.profile import Profile
from widgets.status_widget import StatWidget
from widgets.agi_widget import AGIWidget
from widgets.filter_widget import FilterWidget
from widgets.label_widget import LabelWidget
import qdarkstyle


WIDTH_RIGHT = 250
FONT_SIZE = 8

class SVF(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        font_size = FONT_SIZE
        Profile.set_profile(DATASET_NAME, "example.log")

        # agi
        self.drawer = AGIWidget()

        # statWidget to show status of the graph
        N, E, _ = Profile.get_profile()
        logging.info("Number of nodes: {0}".format(N))
        logging.info("Number of edges: {0}".format(E))
        self.stat_widget = StatWidget(font_size=font_size)

        # labelWidget to show vertice's labels
        self.label_widget = LabelWidget(font_size=font_size)

        # filtering
        self.filter_widget = FilterWidget()
        self.combo_node_centrality = QComboBox()
        self.combo_node_centrality.setFont(QFont("Arial", font_size, QFont.Bold))
        self.combo_node_centrality.setFixedWidth(WIDTH_RIGHT)
        self.combo_node_centrality.addItem("No filter")
        centrality_dir = Profile.dataset_dir.joinpath(Profile.name, "centrality/v")
        for c_file in os.listdir(str(centrality_dir)):
            name_npy = c_file.split('.')
            if len(name_npy) == 2 and name_npy[1] == "npy":
                self.combo_node_centrality.addItem(name_npy[0])
        self.combo_node_centrality.move(50,50)

        # history buttons
        style = self.app.style()
        left_icon = style.standardIcon(QStyle.StandardPixmap(QStyle.SP_ArrowLeft))
        right_icon = style.standardIcon(QStyle.StandardPixmap(QStyle.SP_ArrowRight))
        history_button_backward = QPushButton()
        history_button_forward = QPushButton()
        history_button_backward.setIcon(left_icon)
        history_button_forward.setIcon(right_icon)

        horizontalLine1 = QWidget()
        horizontalLine1.setFixedSize(WIDTH_RIGHT, 2)
        horizontalLine1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        horizontalLine1.setStyleSheet("background-color: #000000")
        horizontalLine2 = QWidget()
        horizontalLine2.setFixedSize(WIDTH_RIGHT, 2)
        horizontalLine2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        horizontalLine2.setStyleSheet("background-color: #000000")

        filter_layout = QVBoxLayout()
        filter_layout.addLayout(self.stat_widget.layout)
        filter_layout.addWidget(history_button_forward)
        filter_layout.addWidget(history_button_backward)
        filter_layout.addWidget(horizontalLine1)
        filter_label = QLabel("Filter:")
        filter_label.setFont(QFont("Arial", font_size, QFont.Bold))
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.combo_node_centrality)
        filter_layout.addWidget(self.filter_widget)
        filter_layout.addLayout(self.stat_widget.layout)
        newline2 = QLabel("\n")
        newline2.setFont(QFont("Arial", 1))
        filter_layout.addWidget(newline2)
        filter_layout.addWidget(horizontalLine2)
        filter_layout.addLayout(self.label_widget.layout)

        # embed to mainLayout
        mainLayout = QHBoxLayout()
        mainLayout.setAlignment(Qt.AlignTop)
        mainLayout.addWidget(self.drawer)
        mainLayout.addLayout(filter_layout)
        self.setLayout(mainLayout)
        self.setWindowTitle("Social Viewpoint Finder")

        # signals and slots
        self.combo_node_centrality.activated[str].connect(self.update_centrality)
        self.filter_widget.histogram_updated.connect(self.drawer.update_thresholds)
        self.drawer.fps_update.connect(self.stat_widget.setFPS)
        self.drawer.node_clicked.connect(self.label_widget.setLabels)
        self.stat_widget.slider_pf.valueChanged[int].connect(self.drawer.update_pf)
        history_button_backward.clicked.connect(self.drawer.browse_backward)
        history_button_forward.clicked.connect(self.drawer.browse_forward)


    def update_centrality(self, name):
        if name == "No filter":
            self.filter_widget.setDisabled(True)
        else:
            self.filter_widget.setDisabled(False)
        self.drawer.update_centrality(name)

    def on_tick(self):
        pass

    @classmethod
    def start(cls, fullscreen=False, timeout=1000/500):
        cls.app = Application()
        cls.app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        widget = cls()
        widget.move(650, 50)
        widget.show()
        cls.app.startTimer(timeout=timeout, on_tick=widget.drawer.on_tick)
        if fullscreen:
            widget.windowHandle().setVisibility(QWindow.FullScreen)
        cls.app.run()

    def keyPressEvent(self, ev):
        k = ev.key()
        if k == Qt.Key_Escape or k == Qt.Key_Q:
            self.close()
            try:
                sys.exit(self.app.exec_())
            except:
                print("exiting")
        if k == Qt.Key_S:
            logging.info("History of draggings is saved")
            self.drawer.save_record()

    # def closeEvent(self, event):
    #     close = QMessageBox()
    #     close.setText('Quit?')
    #     close.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
    #     close = close.exec()
    #
    #     if close == QMessageBox.Yes:
    #         event.accept()
    #     else:
    #         event.ignore()

if __name__ == '__main__':

    SVF.start(fullscreen=False)

