from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import numpy as np


class FilterWidget(QGraphicsView):
    # signals
    histogram_updated = pyqtSignal(int, int)

    def __init__(self, height=25, width=210):
        super(FilterWidget, self).__init__()
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.p_scene = QGraphicsScene()
        self.setScene(self.p_scene)
        #self.p_scene.setSceneRect(0, 0, 210, 50)

        self.width = width
        self.height = height
        self.size = [210, 25]
        self.setFixedSize(*self.size)

        self.hist = None  # ヒストグラムの分布のリスト
        self.diff = None  # ヒストグラムの目盛り
        self.b1 = 0       # 選択する中心性の境界(下限)
        self.b2 = width   # 選択する中心性の境界(上限)

        self.lines = []  # 中心性の境界線
        self.rects = []  # ヒストグラムの棒

        self.updateHist()
        self.initialize()

        self.text = ""
        self.updated = False

    def initialize(self):
        self.lines.clear()
        self.rects.clear()
        self.p_scene.clear()

        length = self.width / len(self.hist)          # ヒストグラムのバー１本の幅
        vals_fixed = self.hist / max(self.hist) * 40  # ヒストグラムの値を描画するために修正

        for i in range(len(self.hist)):
            pos_x = length * i
            item = QGraphicsRectItem(pos_x, 0, length, 50)
            item.setPen(QPen(Qt.white, 0))
            item.setBrush(QBrush(Qt.white))
            self.p_scene.addItem(item)
            self.rects.append(item)

        self.b1 = 0 ; self.b2 = self.width
        l1 = QGraphicsLineItem(self.b1, 0, self.b1, 50)
        l1.setPen(QPen(Qt.white, 1))
        l2 = QGraphicsLineItem(self.b2, 0, self.b2, 50)
        l2.setPen(QPen(Qt.white, 1))
        self.p_scene.addItem(l1)
        self.lines.append(l1)
        self.p_scene.addItem(l2)
        self.lines.append(l2)
        self.setBoundary(0)
        self.setDisabled(True)

    def updateHist(self):
        self.diff = 10
        self.hist = np.ones(100) * 10
        self.initialize()

    def setBoundary(self, x):
        # ヒストグラムの値域の境界を更新する
        diff = abs(self.b1 - x) < abs(self.b2 - x)
        if diff: # b1のが近い
            if x > 0:
                self.b1 = x
                self.lines[0].setLine(x,0,x,50)
            else:
                self.b1 = 0
                self.lines[0].setLine(0, 0, 0, 50)
        else: # b2のが近い
            if x < self.width :
                self.b2 = x
                self.lines[1].setLine(x,0,x,50)
            else:
                self.b2 = self.width
                self.lines[1].setLine(self.width, 0, self.width, 50)

        self.setParam()

    def setParam(self):
        fill_gray = True
        y = self.rects[0].rect().y() + self.rects[0].rect().height()
        for r in self.rects:
            if fill_gray:
                if r.contains(QPoint(self.b1, y)) and r.contains(QPoint(self.b2, y)):
                    fill_gray = True
                    r.setBrush(Qt.white)
                    r.setPen(QPen(Qt.white, 0))
                elif r.contains(QPoint(self.b1, y)):
                    fill_gray = False
                    r.setBrush(Qt.white)
                    r.setPen(QPen(Qt.white, 0))
                else:
                    r.setBrush(Qt.black)
                    r.setPen(QPen(Qt.black, 0))
            else:
                if r.contains(QPoint(self.b2, y)): fill_gray = True
                r.setBrush(Qt.white)
                r.setPen(QPen(Qt.white, 0))
        self.update()
        return 0

    def mousePressEvent(self, event):
        delta = (self.size[0] - self.width) * 0.5
        x = event.pos().x() - int(delta)
        self.setBoundary(x)
        super(FilterWidget, self).mousePressEvent(event)
        self.histogram_updated.emit(int(self.b1 * 100 / self.width), int(self.b2 * 100 / self.width))
        self.text = str(x)

    def mouseMoveEvent(self, event):
        delta = (self.size[0] - self.width) * 0.5
        x = event.pos().x() - int(delta)
        self.setBoundary(x)
        super(FilterWidget, self).mouseMoveEvent(event)
        self.histogram_updated.emit(int(self.b1 * 100 / self.width), int(self.b2 * 100 / self.width))
        self.text = str(x)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    mainWindow = FilterWidget()
    mainWindow.show()
    sys.exit(app.exec_())
