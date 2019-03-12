from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.Qt import QFont


class DemoWidget(QWidget):
    def __init__(self, W=210, font_size=8):
        super().__init__()
        self.setFixedWidth(W)
        self.layout = QVBoxLayout()
        self.radiobuttons = {}
        self.initUI(W, font_size)

    def initUI(self, W, font_size):
        # select the network to visualize
        lbl_selection = QLabel("Select the network!")
        lbl_selection.setFont(QFont("Arial", font_size, QFont.Bold))
        self.radiobuttons = {}
        for s in 'lesmis polblogs 4dai_uni math math_without_hub'.split(' '):
            qrb = QRadioButton()
            qrb.setText(s)
            qrb.setCheckable(True)
            qrb.setFocusPolicy(Qt.NoFocus)
            self.radiobuttons.update({s: qrb})

        self.radiobuttons["math"].setChecked(True)

        self.group = QButtonGroup()
        for idx, (lbl, button) in enumerate(self.radiobuttons.items()):
            self.group.addButton(button, idx+1)

        toggle_layout = QVBoxLayout()
        # toggle_layout.addWidget(lbl_selection)
        for _, button in self.radiobuttons.items():
            toggle_layout.addWidget(button)

        # qrcode の画像
        qr_img = QPixmap("img/vimeo_iui18.png")
        qrcode = QLabel(self)
        qrcode.setPixmap(qr_img)

        self.layout.addLayout(toggle_layout)
        self.layout.addWidget(qrcode)
        self.setLayout(self.layout)

