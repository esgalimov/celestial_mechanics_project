import sys
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QDialog
from PyQt5 import QtCore, QtGui, QtWidgets
import scipy.constants as constants
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import sqlite3
import gc
from error import Ui_Error
from settings import Ui_Settings
from window import Ui_MainWindow
from window_about import Ui_About
from window_added import Ui_Added


# словарь со звездами (массы в солнечных массах)
STARS_DICT = {'Солнце': 1, 'Альфа Центавра': 0.9, 'Бетельгейзе': 11,
              'Сириус': 2, 'Арктур': 1.3, 'Ригель': 18, 'Альдебаран': 2.5}
# астрономическая единица в метрах
A_E = 1.5 * (10 ** 11)
# масса солнца в килограммах
SOLAR_MASS = 1.989 * (10 ** 30)
# грав. постоянная
G = constants.G
# промежуток времени
DT = 30
# кол-во точек
N = 2000000
INF = 10 ** 16


# 2D
@njit(fastmath=True)
def count_2d(x0, y0, v0, alpha, M, n, dt, k1, k):
    stop_count = 0
    alpha0 = (alpha / 180) * np.pi
    x = np.zeros(n)
    x[0] = x0
    y = np.zeros(n)
    y[0] = y0
    vx = np.zeros(n)
    vx[0] = v0 * np.cos(alpha0)
    vy = np.zeros(n)
    vy[0] = v0 * np.sin(alpha0)
    for i in range(1, n):
        if (x[i - 1] == 0 and y[i - 1] == 0) or (x[i - 1] > INF or y[i - 1] > INF):
            stop_count = i - 1
            break

        ax = ((-G * M * x[i - 1]) / ((x[i - 1] ** 2 + y[i - 1] ** 2) ** 1.5)) + (k - k1) * vx[i - 1]
        vx2 = vx[i - 1] + dt * ax
        ax1 = ((-G * M * (x[i - 1] + vx[i - 1] * dt)) / (
                    ((x[i - 1] + vx[i - 1] * dt) ** 2 + (y[i - 1] + vy[i - 1] * dt) ** 2) ** 1.5)) + (k - k1) * vx2
        vx[i] = vx[i - 1] + (dt / 2) * (ax + ax1)
        x[i] = x[i - 1] + (dt / 2) * (vx[i - 1] + vx[i])

        ay = ((-G * M * y[i - 1]) / ((x[i - 1] ** 2 + y[i - 1] ** 2) ** 1.5)) + (k - k1) * vy[i - 1]
        vy2 = vy[i - 1] + dt * ay
        ay1 = ((-G * M * ((y[i - 1]) + vy[i - 1] * dt)) / (
                    ((x[i - 1] + vx[i - 1] * dt) ** 2 + (y[i - 1] + vy[i - 1] * dt) ** 2) ** 1.5)) + (k - k1) * vy2
        vy[i] = vy[i - 1] + (dt / 2) * (ay + ay1)
        y[i] = y[i - 1] + (dt / 2) * (vy[i - 1] + vy[i])
    if stop_count:
        for i in range(stop_count, len(x) - 1):
            x[i] = x[stop_count]
            y[i] = y[stop_count]
    return x, y, vx, vy


# 3D
@njit(fastmath=True)
def count_3d(x0, y0, z0, vx0, vy0, vz0, M, n, dt, k1, k):
    stop_count = 0
    x = np.zeros(n)
    x[0] = x0
    y = np.zeros(n)
    y[0] = y0
    z = np.zeros(n)
    z[0] = z0
    vx = np.zeros(n)
    vx[0] = vx0
    vy = np.zeros(n)
    vy[0] = vy0
    vz = np.zeros(n)
    vz[0] = vz0

    for i in range(1, n):
        if (x[i - 1] == 0 and y[i - 1] == 0 and z[i - 1] == 0) or (x[i - 1] > INF or y[i - 1] > INF or z[i - 1] > INF):
            stop_count = i - 1
            break
        ax = ((-G * M * x[i - 1]) / ((x[i - 1] ** 2 + y[i - 1] ** 2 + z[i - 1] ** 2) ** 1.5)) + (k - k1) * vx[i - 1]
        vx2 = vx[i - 1] + dt * ax
        ax1 = ((-G * M * (x[i - 1] + vx[i - 1] * dt)) / (((x[i - 1] + vx[i - 1] * dt) ** 2 + (
                    y[i - 1] + vy[i - 1] * dt) ** 2 + (z[i - 1] + vz[i - 1] * dt) ** 2) ** 1.5)) + (k - k1) * vx2
        vx[i] = vx[i - 1] + (dt / 2) * (ax + ax1)
        x[i] = x[i - 1] + (dt / 2) * (vx[i - 1] + vx[i])

        ay = ((-G * M * y[i - 1]) / ((x[i - 1] ** 2 + y[i - 1] ** 2 + z[i - 1] ** 2) ** 1.5)) + (k - k1) * vy[i - 1]
        vy2 = vy[i - 1] + dt * ay
        ay1 = ((-G * M * (y[i - 1] + vy[i - 1] * dt)) / (((x[i - 1] + vx[i - 1] * dt) ** 2 + (
                    y[i - 1] + vy[i - 1] * dt) ** 2 + (z[i - 1] + vz[i - 1] * dt) ** 2) ** 1.5)) + (k - k1) * vy2
        vy[i] = vy[i - 1] + (dt / 2) * (ay + ay1)
        y[i] = y[i - 1] + (dt / 2) * (vy[i - 1] + vy[i])

        az = ((-G * M * z[i - 1]) / ((x[i - 1] ** 2 + y[i - 1] ** 2 + z[i - 1] ** 2) ** 1.5)) + (k - k1) * vz[i - 1]
        vz2 = vz[i - 1] + dt * az
        az1 = ((-G * M * (z[i - 1] + vy[i - 1] * dt)) / (((x[i - 1] + vx[i - 1] * dt) ** 2 + (
                    y[i - 1] + vy[i - 1] * dt) ** 2 + (z[i - 1] + vz[i - 1] * dt) ** 2) ** 1.5)) + (k - k1) * vz2
        vz[i] = vz[i - 1] + (dt / 2) * (az + az1)
        z[i] = z[i - 1] + (dt / 2) * (vz[i - 1] + vz[i])
    if stop_count:
        for i in range(stop_count, len(x) - 1):
            x[i] = x[stop_count]
            y[i] = y[stop_count]
            z[i] = z[stop_count]
    return x, y, z, vx, vy, vz


# график 2D
def draw_2d(to_draw):
    data_arrays = np.array(to_draw)
    x_lim1 = []
    x_lim2 = []
    y_lim1 = []
    y_lim2 = []

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # print(data_arrays)
    for j in data_arrays:
        x1 = np.zeros(int(N / 1000))
        y1 = np.zeros(int(N / 1000))

        for i in range(0, int(N / 1000)):
            x1[i] = j[0][i * 1000]

        for i in range(0, int(N / 1000)):
            y1[i] = j[1][i * 1000]

        xmax = max(x1)
        ymax = max(y1)
        maxm = max([xmax, ymax])

        xmin = min(x1)
        ymin = min(y1)
        minm = min([xmin, ymin])

        deltax = (maxm - xmax + minm - xmin) / 2
        deltay = (maxm - ymax + minm - ymin) / 2

        x_lim1.append(minm - deltax)
        x_lim2.append(maxm - deltax)

        y_lim1.append(minm - deltay)
        y_lim2.append(maxm - deltay)

        ax.plot(x1, y1)

    ax.set_xlim(min(x_lim1), max(x_lim2))
    ax.set_ylim(min(y_lim1), max(y_lim2))

    ax.scatter(0, 0, color='red')
    ax.grid(True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


# график 3D
def draw_3d(to_draw):
    data_arrays = np.array(to_draw)
    x_lim1 = []
    x_lim2 = []
    y_lim1 = []
    y_lim2 = []
    z_lim1 = []
    z_lim2 = []

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for j in data_arrays:
        x1 = np.zeros(int(N / 1000))
        y1 = np.zeros(int(N / 1000))
        z1 = np.zeros(int(N / 1000))

        for i in range(0, int(N / 1000)):
            x1[i] = j[0][i * 1000]

        for i in range(0, int(N / 1000)):
            y1[i] = j[1][i * 1000]

        for i in range(0, int(N / 1000)):
            z1[i] = j[2][i * 1000]

        xmax = max(x1)
        ymax = max(y1)
        zmax = max(z1)

        xmin = min(x1)
        ymin = min(y1)
        zmin = min(z1)

        minm = min([xmin, ymin, zmin])
        maxm = max([xmax, ymax, zmax])

        deltax = (maxm - xmax + minm - xmin) / 2
        deltay = (maxm - ymax + minm - ymin) / 2
        deltaz = (maxm - zmax + minm - zmin) / 2

        x_lim1.append(minm - deltax)
        x_lim2.append(maxm - deltax)

        y_lim1.append(minm - deltay)
        y_lim2.append(maxm - deltay)

        z_lim1.append(minm - deltaz)
        z_lim2.append(maxm - deltaz)
        ax.plot(x1, y1, z1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter(0, 0, 0, color='red')
    ax.set_xlim(min(x_lim1), max(x_lim2))
    ax.set_ylim(min(y_lim1), max(y_lim2))
    ax.set_zlim(min(z_lim1), max(z_lim2))
    plt.show()


# окно 'О программе'
class AboutWindow(QWidget, Ui_About):
    def __init__(self):
        super(AboutWindow, self).__init__()
        self.setupUi(self)


class AddedWindow(QWidget, Ui_Added):
    def __init__(self):
        super(AddedWindow, self).__init__()
        self.setupUi(self)


# окно ошибки
class ErrorWidget(QDialog, Ui_Error):
    def __init__(self):
        super().__init__()
        self.setupUi(self)


# окно настроек
class SettingsWindow(QWidget, Ui_Settings):
    def __init__(self):
        global N, DT
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.new_settings)
        self.pushButton_2.clicked.connect(self.close)
        self.error = ErrorWidget()
        self.line_n.setText(str(N))
        self.line_dt.setText(str(DT))

    def new_settings(self):
        global N, DT
        try:
            N = int(self.line_n.text())
            DT = int(self.line_dt.text())
        except ValueError:
            self.error.show()


class MyWidget(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.run)
        self.add_planet.clicked.connect(self.add)
        self.action_discard.triggered.connect(self.discard)
        self.dis_added.clicked.connect(self.discard_added)
        # списки полей ввода
        self.fields = [self.velocity, self.x_coord,
                       self.y_coord, self.angle, self.koeff, self.koeff1, self.star_mass]
        self.fields_3d = [self.x_3d, self.y_3d, self.z_3d,
                          self.vx_3d, self.vy_3d, self.vz_3d,
                          self.koeff_3d, self.koeff1_3d]

        self.solar_system_planets = [self.earth, self.mercury, self.venus,
                                     self.mars, self.jupiter, self.saturn,
                                     self.uranus, self.neptune, self.pluto]
        for planet in self.solar_system_planets:
            planet.triggered.connect(self.solar_system_input)

        for field in self.fields[:-1]:
            field.setText('0')
        for field in self.fields_3d:
            field.setText('0')
        self.star_mass_box.addItems(STARS_DICT.keys())
        self.star_mass_box.activated.connect(self.star_mass_box_func)
        self.star_mass_box_func()
        # окна
        self.about_window = AboutWindow()
        self.added_window = AddedWindow()
        self.error_widget = ErrorWidget()
        self.settings_window = SettingsWindow()

        self.action_about.triggered.connect(self.about)
        self.action_settings.triggered.connect(self.settings)
        self.action_added.triggered.connect(self.added_show)

        self.to_draw_2d = []
        self.to_draw_3d = []

        self.added_list = {'2d': [], '3d': []}

    def add(self):
        global N, DT
        gc.collect()
        if not plt.get_fignums():
            try:
                if self.tabWidget.currentIndex() == 0:
                    if sum([len(i[0]) + len(i[1]) for i in self.to_draw_2d]) +\
                            sum([len(i[0]) + len(i[1] + len(i[2])) for i in self.to_draw_3d]) + N > 20_000_000:
                        self.error_widget.label.setText('Ошибка: суммарно больше 20 000 000 точек')
                        self.error_widget.show()
                    else:
                        x0 = float(self.x_coord.text()) * A_E
                        y0 = float(self.y_coord.text()) * A_E
                        v0 = float(self.velocity.text())
                        alpha = float(self.angle.text())
                        M = float(self.star_mass.text()) * SOLAR_MASS
                        k = float(self.koeff.text())
                        k1 = float(self.koeff1.text())

                        x, y, vx, vy = count_2d(x0, y0, v0, alpha, M, N, DT, k1, k)
                        self.added_list['2d'].append(f'x0: {self.x_coord.text()} y0: {self.y_coord.text()},'
                                                     f' v0: {self.velocity.text()}, alpha: {self.angle.text()},'
                                                     f' M: {self.star_mass.text()},'
                                                     f' k1: {self.koeff1.text()}, k: {self.koeff.text()}')

                        self.to_draw_2d.append([x, y])

                else:
                    if sum([len(i[0]) + len(i[1] + len(i[2])) for i in self.to_draw_3d]) +\
                            sum([len(i[0]) + len(i[1]) for i in self.to_draw_2d]) + N > 20_000_000:
                        self.error_widget.label.setText('Ошибка: суммарно больше 20 000 000 точек')
                        self.error_widget.show()
                    else:
                        x0 = float(self.x_3d.text()) * A_E
                        y0 = float(self.y_3d.text()) * A_E
                        z0 = float(self.z_3d.text()) * A_E
                        vx0 = float(self.vx_3d.text())
                        vy0 = float(self.vy_3d.text())
                        vz0 = float(self.vz_3d.text())
                        k = float(self.koeff_3d.text())
                        k1 = float(self.koeff1_3d.text())
                        M = float(self.star_mass.text()) * SOLAR_MASS
                        print(x0, y0, z0, vx0, vy0, vz0, k, k1, M)

                        x, y, z, vx, vy, vz = count_3d(x0, y0, z0, vx0, vy0, vz0, M, N, DT, k1, k)
                        self.added_list['3d'].append(f'x0: {self.x_3d.text()} y0: {self.y_3d.text()},'
                                                     f' z0: {self.z_3d.text()}, vx0: {self.vx_3d.text()},'
                                                     f' vy0: {self.vy_3d.text()}, vz0: {self.vz_3d.text()},'
                                                     f' M: {self.star_mass.text()}, k1: {self.koeff1_3d.text()},'
                                                     f' k: {self.koeff_3d.text()}')

                        self.to_draw_3d.append([x, y, z])

            except ValueError:
                self.error_widget.label.setText('Ошибка: неверный формат ввода')
                self.error_widget.show()

            except ZeroDivisionError:
                self.error_widget.label.setText('Ошибка: неверный формат ввода')
                self.error_widget.show()

            except MemoryError:
                self.error_widget.label.setText('Ошибка: перегрузка памяти')
                self.error_widget.show()

    def run(self):
        if not plt.get_fignums():
            if self.tabWidget.currentIndex() == 0:
                if self.to_draw_2d == []:
                    self.add()
                if 0 < len(self.to_draw_2d) <= 3:
                    draw_2d(self.to_draw_2d)
                elif len(self.to_draw_2d) > 3:
                    self.error_widget.label.setText('Ошибка: больше 3 графиков')
                    self.error_widget.show()
            else:
                if self.to_draw_3d == []:
                    self.add()
                if 0 < len(self.to_draw_3d) <= 3:
                    draw_3d(self.to_draw_3d)
                elif len(self.to_draw_2d) > 3:
                    self.error_widget.label.setText('Ошибка: больше 3 графиков')
                    self.error_widget.show()

    def about(self):
        self.about_window.show()

    def settings(self):
        self.settings_window.show()

    def added_show(self):
        self.added_window.show()
        self.added_window.textEdit.clear()
        self.added_window.textEdit.append('2D')
        for i in self.added_list['2d']:
            self.added_window.textEdit.append(i)
        self.added_window.textEdit.append('3D')
        for i in self.added_list['3d']:
            self.added_window.textEdit.append(i)

    def discard(self):
        self.star_mass_box_func()
        for field in self.fields[:-1]:
            field.setText('0')
        for field in self.fields_3d:
            field.setText('0')

    def discard_added(self):
        self.to_draw_2d.clear()
        self.to_draw_3d.clear()
        self.added_list['2d'].clear()
        self.added_list['3d'].clear()

    def star_mass_box_func(self):
        self.star_mass.setText(str(STARS_DICT[self.star_mass_box.currentText()]))

    def solar_system_input(self):
        planet = self.solar_system_planets.index(self.sender()) + 1
        con = sqlite3.connect('planets_db.db')
        cur = con.cursor()
        result = cur.execute('select * from planets where id = ?', (planet, )).fetchone()
        self.velocity.setText(str(result[4]))
        self.x_coord.setText(str(result[2]))
        self.y_coord.setText(str(result[3]))
        self.angle.setText(str(result[5]))
        self.star_mass.setText(str(result[6]))


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = MyWidget()
    form.show()
    sys.excepthook = except_hook
    sys.exit(app.exec())
