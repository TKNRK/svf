import time
import logging
import pickle
from ctypes import *
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import pyqtSignal
from sn.qt import *
from sn.gl import *
import sn.gl.geometry.t3d as T
from constants import *
from ssb import *
from svf import agi
from svf import filtering
from svf.profile import Profile

logging.getLogger().setLevel(logging.INFO)
# Profile.set_profile(DATASET_NAME, "example.log")
number_of_nodes, number_of_edges, dim_hd = Profile.get_profile()
# Determine the maximum size of SSB
MAX_N = number_of_nodes
MAX_E = number_of_edges
MAX_DIM = dim_hd


class AGIWidget(GLWidget):
    fps_update = pyqtSignal(int)
    node_clicked = pyqtSignal(int, list, list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.W, self.H = 700, 500
        self.magnitude = 0.8
        self.setMinimumSize(self.W, self.H)

        # プログラムとか
        self.graphics = None
        self.compute = None

        # フィルタリング情報 (threshold)
        self.threshold_min = -1
        self.delta = 0

        # subroutine
        self.subroutine_vs = dict()
        self.subroutine_fs = dict()
        self.subroutine_cs = dict()
        '@type: Dict[str, int]'

        # mouse action
        self.clicked_pos = None
        self.pick_buf = None
        self.pick_id = -1

        # Benchmark
        # self.FPS = []
        self.renderSec = 0

        # projection factors:
        self.initial_pf = None
        self.projections = None
        self.alpha = PROJECTION_FACTOR

        # history
        self.history = []
        self.history_pointer = 0
        self.history_video = []  # history of drag events, ith element shows an animation between ith and i+1th histories
        self.history_temp = None
        self.history_picked = []

    def minimumSizeHint(self):
        return QtCore.QSize(200, 200)

    def resizeGL(self, w: int, h: int):
        super().resizeGL(w, h)
        self.W = w
        self.H = h
        self.graphics.use()
        self.graphics.u['P'](T.perspective(45., w/float(h), 0.1, 1000.))

    def initializeGL(self):
        # グラフ描画情報
        _, self.initial_pf, self.layout_hd, \
        edges, self.edge_type, _, _, _ = agi.loadgraph(pf=PROJECTION_FACTOR)
        self.projections = agi.gram_schmidt(self.initial_pf ** self.alpha)
        self.history.append(self.projections.copy())

        super().initializeGL()
        self.graphics = Program('widgets/shaders/main.shaders')
        self.compute = Program('widgets/shaders/compute.cs')

        self.graphics.use()
        VertexArray().bind()

        # ユニフォーム変数の送付
        self.Model = np.eye(4, dtype=np.float32)
        eye, target, up = T.vec3(0, 0, 4), T.vec3(0, 0, 0), T.vec3(0, 1, 0)
        View = T.lookat(eye, target, up)
        self.graphics.u['MV'](View.dot(self.Model))
        self.graphics.u['N'](number_of_nodes)
        self.graphics.u['transparency'](0.5)

        # シェーダーのバッファー
        [self.pick_buf, self.lay_buf, self.pro_buf, self.pos_buf, self.filter_buf, self.flag_buf, self.edge_buf, self.drawnBuf, isoBuf] = glGenBuffers(9)

        # マウスアクションの SSB の準備と初期化
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.pick_buf)
        ssb_pick = SSB_Picker()
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(ssb_pick), pointer(ssb_pick), GL_DYNAMIC_READ)

        # 高次元配置情報の SSB の準備と初期化
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.lay_buf)
        ssb_lay = SSB_Layout()
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(ssb_lay), pointer(ssb_lay), GL_STATIC_COPY)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.lay_buf)
        # 高次元配置をSSBに書き込み
        ssb_lay = cast(glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY), POINTER(SSB_Layout)).contents
        layout_hd = self.layout_hd.flatten()
        for i in range(number_of_nodes * dim_hd):
            ssb_lay.layout_hd[i] = layout_hd[i]
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        # 射影ベクトル情報の SSB の準備と初期化
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self.pro_buf)
        ssb_proj = SSB_Projection()
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(ssb_proj), pointer(ssb_proj), GL_DYNAMIC_COPY)
        self.set_ssb_projection()

        # 頂点情報の SSB の準備と初期化
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self.pos_buf)
        ssb_pos = SSB_Position()
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(ssb_pos), pointer(ssb_pos), GL_DYNAMIC_DRAW)

        # フィルターに用いる頂点の中心性スコアの準備と初期化
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, self.filter_buf)
        ssb_filter = SSB_Filter()
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(ssb_filter), pointer(ssb_filter), GL_DYNAMIC_COPY)

        # 描画する色を管理する
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, self.flag_buf)
        ssb_flag = SSB_Flags()
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(ssb_flag), pointer(ssb_flag), GL_DYNAMIC_COPY)

        # 頂点配列の設定
        glBindBuffer(GL_ARRAY_BUFFER, self.pos_buf)
        pos_vs = self.graphics.a['pos_vs'].loc
        glVertexAttribPointer(pos_vs, 3, GL_FLOAT, GL_FALSE, 12, None)
        glEnableVertexAttribArray(pos_vs)

        # 辺をSSBに保存する設定
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, self.edge_buf)
        ssb_edge = SSB_Edgelist()
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(ssb_edge), pointer(ssb_edge), GL_STATIC_DRAW)
        # SSBに書き込み
        ssb_edge = cast(glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY), POINTER(SSB_Edgelist)).contents
        for i in range(number_of_edges * 2):
            ssb_edge.edges[i] = edges[i]
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        # 辺のバインド
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.edge_buf)

        # フィルター情報を保存
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, self.drawnBuf)
        ssb_drawn = SSB_isDrawn()
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(ssb_drawn), pointer(ssb_drawn), GL_DYNAMIC_COPY)

        # 孤立点情報を保持
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, isoBuf)
        ssb_iso = SSB_Isolation()
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(ssb_iso), pointer(ssb_iso), GL_DYNAMIC_COPY)

        # Enable とか
        glDisable(GL_DEPTH_TEST)
        for p in [GL_VERTEX_PROGRAM_POINT_SIZE, GL_BLEND]:
            glEnable(p)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # set background color
        # glClearColor(0.2, 0.2, 0.2, 1.0)

        # set subroutines
        for subroutine in ["draw_vertices", "draw_edges"]:
            self.subroutine_vs[subroutine] = glGetSubroutineIndex(self.graphics._program, GL_VERTEX_SHADER, subroutine)
            self.subroutine_fs[subroutine] = glGetSubroutineIndex(self.graphics._program, GL_FRAGMENT_SHADER, subroutine)
        self.subroutine_fs["pick"] = glGetSubroutineIndex(self.graphics._program, GL_FRAGMENT_SHADER, "pick")

        self.compute.use()
        self.compute.u['dim_hd'](dim_hd)
        self.compute.u['magnitude'](self.magnitude)
        # set subroutines
        for subroutine in ["proj", "_find_neighbors", "find_neighbors", "clear_annotation", "NoFilter", "filtering", "find_isolated"]:
            self.subroutine_cs[subroutine] = glGetSubroutineIndex(self.compute._program, GL_COMPUTE_SHADER, subroutine)

        # initialize the layout
        glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, self.subroutine_cs["proj"])
        glDispatchCompute(int(number_of_nodes / 1000) + 1, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        # initialize the filter settings
        self.update_centrality("No filter")

    def load_data(self):
        # get information of the new network
        global number_of_nodes, number_of_edges, dim_hd
        number_of_nodes, number_of_edges, dim_hd = Profile.get_profile()
        # グラフ描画情報
        _, self.initial_pf, self.layout_hd, \
        edges, self.edge_type, _, _, _ = agi.loadgraph(pf=PROJECTION_FACTOR)
        self.projections = agi.gram_schmidt(self.initial_pf ** self.alpha)

        # Graphics Pipeline: replace the data on SSB
        self.graphics.use()
        self.graphics.u['N'](number_of_nodes)
        self.graphics.u['transparency'](0.5)
        # update the projection matrix
        self.set_ssb_projection()
        # update the high-dimensional layout
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.lay_buf)
        ssb_lay = cast(glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY), POINTER(SSB_Layout)).contents
        layout_hd = self.layout_hd.flatten()
        for i in range(number_of_nodes * dim_hd):
            ssb_lay.layout_hd[i] = layout_hd[i]
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        # update the edgelist
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.edge_buf)
        ssb_edge = cast(glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY), POINTER(SSB_Edgelist)).contents
        for i in range(number_of_edges * 2):
            ssb_edge.edges[i] = edges[i]
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        # Compute shader: initialize the vertex coordinates
        self.compute.use()
        self.compute.u['dim_hd'](dim_hd)
        self.magnitude = 0.8
        self.compute.u['magnitude'](self.magnitude)
        # initialize the layout
        glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, self.subroutine_cs["proj"])
        glDispatchCompute(int(number_of_nodes / 1000) + 1, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        # CPU-side Application: initialize the UI
        # initialize the filter event
        self.update_centrality("No filter")
        # initialize the picker event
        self.should_handle_pick(pos=QtCore.QPoint(0,0))

    def set_ssb_projection(self):
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.pro_buf)
        ssb = cast(glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY), POINTER(SSB_Projection)).contents
        projection = self.projections.flatten()
        for i in range(dim_hd * 2):
            ssb.projection[i] = projection[i]
        # ssb.projection = projection.ctypes.data_as(ctypes.POINTER(c_float * 1000))
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def set_ssb_centrality(self, centrality):
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.filter_buf)
        ssb = cast(glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY), POINTER(SSB_Filter)).contents
        for i in range(len(centrality)):
            ssb.score[i] = centrality[i]
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def get_neighbors(self):
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.flag_buf)
        ssb = cast(glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY), POINTER(SSB_Flags)).contents
        flags = np.zeros(number_of_nodes)
        for i in range(number_of_nodes):
            flags[i]= ssb.vertexColor[i]
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        flags_sorted = flags.argsort()
        indices_neighbor = np.where((flags_sorted == self.pick_id) == True)
        if indices_neighbor[0].shape == 0:
            return set()
        else:
            return set(flags_sorted[indices_neighbor[0][0]:][1:])

    def get_filtered(self):
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.drawnBuf)
        ssb = cast(glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY), POINTER(SSB_isDrawn)).contents
        isDrawn = np.zeros(number_of_nodes)
        for i in range(number_of_nodes):
            isDrawn[i]= ssb.isDrawn[i]
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        print(isDrawn)
        hidden = np.where(isDrawn == 0)
        if hidden[0].shape == 0:
            return set()
        else:
            return set(hidden[0])

    def paintGL(self):
        t = Time.time
        # FPS の計測
        if t > self.nextFPS and self.fps != 0:
            self.fps_update.emit(self.fps)
            # self.FPS.append(self.fps)
            # logging.info('Average FPS / {0} times: {1}'.format(len(self.FPS), np.mean(self.FPS)))
            # logging.info('Average render time: {0}'.format(self.renderSec / (self.fps) * 1e3))
            self.renderSec = 0
        super().paintGL()

        s = time.time()
        self.graphics.use()
        if self.should_handle_pick():
            # pick の動作を行う
            self.handle_pick_before()
            glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, self.subroutine_vs['draw_vertices'])
            glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, self.subroutine_fs['pick'])
            glDrawArrays(GL_POINTS, 0, number_of_nodes)
            self.handle_pick_after()

        # 辺の描画
        glDisable(GL_DEPTH_TEST)
        glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, self.subroutine_vs['draw_edges'])
        glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, self.subroutine_fs['draw_edges'])
        glDrawElements(GL_LINES, number_of_edges * 2, self.edge_type, ctypes.c_void_p(0))
        # 頂点の描画
        glEnable(GL_DEPTH_TEST)
        glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, self.subroutine_vs['draw_vertices'])
        glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, self.subroutine_fs['draw_vertices'])
        glDrawArrays(GL_POINTS, 0, number_of_nodes)
        glFinish()
        e = time.time()
        self.renderSec += e-s

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        super(AGIWidget, self).mousePressEvent(ev)
        self.should_handle_pick(pos=ev.pos())
        self.pick_id = -1
        # prepare for recording the history
        self.history_temp = []

    def should_handle_pick(self, pos=None):
        should_handle = self.clicked_pos is not None
        if not should_handle and pos is not None:
            logging.debug('should handle pick ...')
            self.clicked_pos = pos
            should_handle = True
        return should_handle

    def handle_pick_before(self):
        logging.debug('before - lock ssb')
        logging.debug('before - glBindBuffer')
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.pick_buf)
        logging.debug('before - glMapBuffer')
        # Map the GPU-side shader-storage-buffer on the application, allowing for write-only access
        ssb = cast(glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY), POINTER(SSB_Picker)).contents
        # Save the clicked location information
        ssb.clicked_x, ssb.clicked_y = self.clicked_pos.x(), self.clicked_pos.y()
        # Initialize fields
        ssb.pick_z = float('-inf')         # Initially -infty
        ssb.pick_lock, ssb.pick_id = 0, -1  # Initially UNLOCKED (c.f., Unlocked@kw4.shader)
        logging.debug('clicked pos: ({}, {})'.format(ssb.clicked_x, ssb.clicked_y))
        # Unmap the SSB
        logging.debug('before - glUnmapBuffer')
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
        # Tell the next rendering cycle to perform pick-identification
        logging.debug('before - glUnbindBuffer')
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def handle_pick_after(self):
        logging.debug('after - glBindBuffer')
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.pick_buf)
        # Map the GPU-side shader-storage-buffer on the application, allowing for read-only access
        ssb = cast(glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY), POINTER(SSB_Picker)).contents
        logging.debug('x: {}, y: {}'.format(ssb.clicked_x, ssb.clicked_y))
        logging.info('id: {} (z: {})'.format(ssb.pick_id, ssb.pick_z))
        self.pick_id = ssb.pick_id
        # Unmap the SSB
        logging.debug('after - glUnmapBuffer')
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
        # Pick-identification finished
        logging.debug('after - glUnbindBuffer')
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        self.clicked_pos = None
        # If picked, color the nodes and show labels. Else, whitening the nodes and delete labels.
        self.compute.use()
        if self.pick_id < 0:
            glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, self.subroutine_cs["clear_annotation"])
            glDispatchCompute(int(number_of_nodes / 1000) + 1, 1, 1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
            self.node_clicked.emit(-1, [], [])
        else:
            # 頂点色フラグの消去と、選択された頂点にフラグを設定
            glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, self.subroutine_cs["_find_neighbors"])
            glDispatchCompute(int(number_of_nodes / 1000) + 1, 1, 1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
            # 隣接頂点の探索
            glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, self.subroutine_cs["find_neighbors"])
            glDispatchCompute(int(number_of_edges / 1000) + 1, 1, 1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
            #　GPUから隣接頂点情報を取得する
            neighbor = self.get_neighbors()
            hidden = self.get_filtered()
            neighbor_appear = neighbor - hidden
            neighbor_hidden = neighbor - neighbor_appear
            self.node_clicked.emit(self.pick_id, list(neighbor_appear), list(neighbor_hidden))
        self.graphics.use()

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        super(AGIWidget, self).mouseMoveEvent(ev)
        # 頂点が選ばれているならば、射影ベクトルを更新する
        if self.pick_id > 0:
            pos = ev.pos()
            x, y = pos.x(), pos.y()
            pos_x, pos_y = x / (self.W / 2) - 1, (self.H - y) / (self.H / 2) - 1
            # projections= self.read_projections()
            picked_hd = self.layout_hd[self.pick_id, :]
            self.projections = agi.get_new_projections(picked_hd, pos_x / self.magnitude, pos_y / self.magnitude, self.projections)
            self.set_ssb_projection()
            self.compute.use()
            glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, self.subroutine_cs["proj"])
            glDispatchCompute(int(number_of_nodes / 1000) + 1, 1, 1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
            self.history_temp.append(self.projections.copy())

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        super(AGIWidget, self).mouseReleaseEvent(ev)
        if len(self.history_temp) > 0:
            if self.history_pointer + 1 != len(self.history):
                self.history = self.history[:self.history_pointer + 1]
                self.history_picked = self.history_picked[:self.history_pointer]
                self.history_video = self.history_video[:self.history_pointer + 1]
            new_history = np.array(self.history_temp)
            self.history.append(new_history[-1])
            self.history_picked.append(self.pick_id)
            self.history_video.append(new_history)
            self.history_pointer += 1

    def read_projections(self):
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.pick_buf)
        ssb = cast(glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY), POINTER(SSB_Projection)).contents
        ssb_projections = [ssb.projection[i] for i in range(dim_hd * 2)]
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        x_proj = ssb_projections[0::2]
        y_proj = ssb_projections[1::2]
        projections = np.array([x_proj, y_proj]).T
        return projections

    def show_picked(self):
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.pick_buf)
        # Map the GPU-side shader-storage-buffer on the application, allowing for read-only access
        ssb = cast(glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY), POINTER(SSB_Picker)).contents
        logging.info('SSB: ({0}, {1})'.format(ssb.clicked_x, ssb.clicked_y))
        logging.info('id: {0} (z: {1})'.format(ssb.pick_id, ssb.pick_z))
        # Unmap the SSB
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
        # Pick-identification finished
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def update_centrality(self, name):
        th1 = 0
        th2 = 1
        if name == "No filter":
            self.set_ssb_centrality(np.zeros(number_of_nodes))
        else:
            centrality, score_min, score_max = filtering.load_centrality(name)
            if centrality is None:
                logging.error("This centrality doesn't exist. :{0}".format(name))
            else:
                self.threshold_min = score_min
                self.delta = (score_max - score_min) / 100
                self.set_ssb_centrality(centrality)
                th1 = score_min
                th2 = score_max
        self.compute.use()
        self.compute.u['th1'](th1)
        self.compute.u['th2'](th2)
        glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, self.subroutine_cs["NoFilter"])
        glDispatchCompute(int(number_of_nodes / 1000) + 1, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def update_thresholds(self, num1, num2):
        th1 = self.threshold_min + self.delta * num1
        th2 = self.threshold_min + self.delta * num2
        self.compute.use()
        self.compute.u['th1'](th1)
        self.compute.u['th2'](th2)
        glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, self.subroutine_cs["filtering"])
        glDispatchCompute(int(number_of_nodes / 1000) + 1, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, self.subroutine_cs["find_isolated"])
        glDispatchCompute(int(number_of_edges / 1000) + 1, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def wheelEvent(self, ev: QtGui.QWheelEvent):
        dlt = ev.angleDelta().y()
        if dlt < 0:  # zoom out
            if self.magnitude > 0.01: self.magnitude -= 0.05
        else:  # zoom in
            if self.magnitude < 5.0: self.magnitude += 0.05
        self.compute.use()
        self.compute.u['magnitude'](self.magnitude)
        glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, self.subroutine_cs["proj"])
        glDispatchCompute(int(number_of_nodes / 1000) + 1, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def update_pf(self, value):
        self.alpha = value / 50.0  # in (0.01, 4.0)
        self.projections = agi.gram_schmidt(self.initial_pf ** self.alpha)
        self.set_ssb_projection()
        self.compute.use()
        glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, self.subroutine_cs["proj"])
        glDispatchCompute(int(number_of_nodes / 1000) + 1, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        self.history = [self.projections.copy()]
        self.history_picked.clear()
        self.history_video.clear()
        self.history_pointer = 0

    def browse_forward(self):
        l = len(self.history)
        if l > 1 and self.history_pointer + 1 < l:
            # self.play_record(direction="forward")
            self.history_pointer += 1
            self.projections = self.history[self.history_pointer]
            self.set_ssb_projection()
            self.compute.use()
            glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, self.subroutine_cs["proj"])
            glDispatchCompute(int(number_of_nodes / 1000) + 1, 1, 1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def browse_backward(self):
        if len(self.history) > 1 and self.history_pointer > 0:
            # self.play_record(direction="backward")
            self.history_pointer -= 1
            self.projections = self.history[self.history_pointer]
            self.set_ssb_projection()
            self.compute.use()
            glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, self.subroutine_cs["proj"])
            glDispatchCompute(int(number_of_nodes / 1000) + 1, 1, 1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def play_record(self, direction="forward"):
        record = self.history_video[self.history_pointer]
        if direction == "backward":
            record = record[::-1]
        for r in record:
            self.projections = r
            self.set_ssb_projection()
            self.compute.use()
            glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, self.subroutine_cs["proj"])
            glDispatchCompute(int(number_of_nodes / 1000) + 1, 1, 1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
            time.sleep(10)

    def save_record(self):
        pth_history = Profile.log_dir.joinpath(Profile.name, 'history.npy')
        pth_picked = Profile.log_dir.joinpath(Profile.name, 'history_picked.npy')
        np.save(str(pth_history), np.array(self.history))
        np.save(str(pth_picked), np.array(self.history_picked))
        # self.history = [self.projections.copy()]
        # self.history_picked.clear()
        # self.history_video.clear()
        # self.history_pointer = 0


if __name__ == '__main__':
    AGIWidget.start()