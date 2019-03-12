from PyQt5.QtWidgets import *
from PyQt5.Qt import QFont
from constants import *
from svf.profile import Profile


class LabelWidget:
    def __init__(self, font_size=8):
        self.labels = Profile.get_labels()
        self.selected_box = QTextEdit()
        self.selected_box.setFont(QFont("Arial", font_size))
        self.selected_box.setFixedSize(210, 30)
        self.neighbor_box = QTextEdit()
        self.neighbor_box.setFont(QFont("Arial", font_size))
        self.neighbor_box.setFixedWidth(210)

        self.layout = QVBoxLayout()
        label_selected = QLabel("Selected Node")
        label_selected.setFont(QFont("Arial", font_size, QFont.Bold))
        self.layout.addWidget(label_selected)
        self.layout.addWidget(self.selected_box)
        label_neighbor = QLabel("Neighbors")
        label_neighbor.setFont(QFont("Arial", font_size, QFont.Bold))
        self.layout.addWidget(label_neighbor)
        self.layout.addWidget(self.neighbor_box)

    def load_data(self):
        self.labels = Profile.get_labels()
        self.selected_box.clear()
        self.neighbor_box.clear()

    def setLabels(self, clicked_id, neighbor_ids, hidden_ids):
        if clicked_id < 0:
            self.selected_box.clear()
            self.neighbor_box.clear()
        else:
            self.selected_box.setText(self.labels[clicked_id])
            appearHTML = '<font color="White">'
            hiddenHTML = '<font color="Gray">'
            endHTML = '</font><br>'
            neighbor_text = ""
            for id in neighbor_ids:
                neighbor_text += appearHTML + self.labels[id] + endHTML
            for id in hidden_ids:
                neighbor_text += hiddenHTML + self.labels[id] + endHTML
            self.neighbor_box.setText(neighbor_text)