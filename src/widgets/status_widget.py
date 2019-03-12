from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.Qt import QFont
from constants import *
from svf.profile import Profile


class StatWidget:
    def __init__(self, font_size=8):
        N, E, _ = Profile.get_profile()
        name = Profile.name
        self.layout = QVBoxLayout()

        self.graph_name = QLabel("Network: " + name)
        self.node_size = QLabel("#Vertices: " + str(N))
        self.edge_size = QLabel("#Edges: " + str(E))
        self.fps = QLabel("FPS: ")
        # set font style
        self.graph_name.setFont(QFont("Arial", font_size, QFont.Bold))
        self.node_size.setFont(QFont("Arial", font_size, QFont.Bold))
        self.edge_size.setFont(QFont("Arial", font_size, QFont.Bold))
        self.fps.setFont(QFont("Arial", font_size, QFont.Bold))
        # control projection factor
        self.slider_pf = QSlider(Qt.Horizontal)
        self.slider_pf.setFixedWidth(210)
        self.slider_pf.setTickInterval(1)
        self.slider_pf.setRange(1, 200)
        self.slider_pf.setValue(25)

        self.layout.addWidget(self.graph_name)
        self.layout.addWidget(self.node_size)
        self.layout.addWidget(self.edge_size)
        self.layout.addWidget(self.fps)
        self.layout.addWidget(self.slider_pf)

    def load_data(self):
        N, E, _ = Profile.get_profile()
        name = Profile.name
        self.graph_name.setText("Network: " + name)
        self.node_size.setText("#Vertices: " + str(N))
        self.edge_size.setText("#Edges: " + str(E))

    def setFPS(self, fps):
        self.fps.setText("FPS: " + str(fps))
