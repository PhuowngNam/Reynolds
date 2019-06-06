import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''

NHẬP SỐ LIỆU ĐẦU VÀO

'''

R = float(input("Nhập chiều dài bán kính ổ  (mm) : "))
L = float(input("Nhập chiều dài ổ  (mm) : "))
N = int(input("Nhập số khoảng chia : "))
M = N
epsilon = float(input("Nhập độ lệch tâm tương đối : "))
n = float(input("Nhập tốc độ quay của trục (vòng/phút) : "))
nuy = float(input("Nhập độ nhớt động lực học (Pa.s) : "))
c = float(input("Nhập khe hở bán kính (mm) : "))
phi = float(input("Nhập góc chất tải (độ) : "))

'''

TÍNH TOÁN TRƯỜNG ÁP SUẤT

'''

R = R / 1000
L = L / 1000
c = c / 1000

X = 2 * np.pi * R
k = 2 * np.pi / N
l = 1 / M

const1 = 1 / (k ** 2)
const2 = 1 / (l ** 2)
const3 = (R / L) ** 2
ms = const1 + const3 * const2

omega = 2 * n * np.pi / 60

# h = C(1 + epsilon*cos(theta))
h_min = c * (1 - epsilon)  # theta = 0
h_max = c * (1 + epsilon)  # theta = pi

# vận tốc dài trên trục
v = omega * R
e = epsilon * c

hs_P = np.zeros((N + 1, M + 1))
Mat_hs = np.zeros(((N - 1) * (M - 1), (N - 1) * (M - 1)))
Mat_hs_vp = []
hs_P.shape, Mat_hs.shape
index = 0
for i in np.arange(1, N):
    for j in np.arange(1, M):
        hs_P = np.zeros((N + 1, M + 1))
        theta = k * i
        H = 1 + epsilon * np.cos(theta)
        Hm = 1 + epsilon * np.cos(theta - k)
        Hp = 1 + epsilon * np.cos(theta + k)
        # dH_dx = -epsilon*np.sin(theta)
        dH_dx = (Hp - Hm) / (2 * k)

        A = (const1 - 1.5 * (1 / k * H) * dH_dx) / ms
        B = (const1 + 1.5 * (1 / k * H) * dH_dx) / ms
        C = const2 * const3 / ms
        D = dH_dx / (ms * H * H * H)

        hs_P[i, j] = -2
        hs_P[i - 1, j] = A
        hs_P[i + 1, j] = B
        hs_P[i, j + 1] = C
        hs_P[i, j - 1] = C

        flat_hs_P = hs_P[1:N, 1:M].reshape(1, -1)
        Mat_hs[index] = Mat_hs[index] + flat_hs_P
        Mat_hs_vp.append(D)
        index += 1

Mat_hs_vp = np.asarray(Mat_hs_vp).reshape(-1, 1)
P = np.linalg.inv(Mat_hs) @ Mat_hs_vp
P = P.reshape(N - 1, M - 1)
zeros_row = np.zeros((1, P.shape[1]))
zeros_col = np.zeros((P.shape[0] + 2, 1))

P = np.concatenate((zeros_row, P, zeros_row), axis=0)
P = np.concatenate((zeros_col, P, zeros_col), axis=1)

"""

VẼ ĐỒ THỊ

"""

chuvi = np.arange(0, X + X / N, X / N)
chieudai = np.arange(0, L + L / M, L / M)

chieudai, chuvi = np.meshgrid(chieudai, chuvi, sparse=True)

p = np.zeros((N + 1, M + 1))
for i in range(N + 1):
    for j in range(M + 1):
        if P[i][j] > 0:
            p[i][j] = P[i][j]

fig = plt.figure(num=2)
ax = fig.add_subplot(111, projection='3d')

ax.plot_wireframe(chieudai, chuvi, P)
plt.suptitle('Trường áp suất ổ đỡ', fontsize=20)

ax.set_xlabel('Z axis')
ax.set_ylabel('X axis')
ax.set_zlabel('P axis')
plt.show()

# cut mat phang

plt.plot(chuvi, P[:, int(N / 2)] * (6 * nuy * omega * ((R / c) ** 2)))
plt.suptitle('Mặt cắt theo phương chu vi', fontsize=20)
plt.xlabel("Chu vi(m)", fontsize=15)
plt.ylabel("Áp suất N", fontsize=15)
plt.grid()
plt.show()

# Chieu cao hi
H = []
delta_theta = 2*np.pi/N
for i in range(1, N):
    hi = c*(1 + epsilon*np.cos(delta_theta*i))
    H.append(hi)

plt.plot(H, 'ro')
plt.suptitle('Chiều dày màng dầu', fontsize=20)
plt.xlabel("Chu vi(m)", fontsize=15)
plt.ylabel("hi", fontsize=15)
plt.show()


"""

TÍNH KHẢ NĂNG TẢI CỦA Ổ VÀ MOMEN MA SÁT

"""


# Lực nâng không thứ nguyên
def tinh_N(si, ni):
    N1 = 0.25 * (1 - si) * (1 - ni)
    N2 = 0.25 * (1 + si) * (1 - ni)
    N3 = 0.25 * (1 + si) * (1 + ni)
    N4 = 0.25 * (1 - si) * (1 + ni)

    return N1, N2, N3, N4


can3_inv = 1 / np.sqrt(3)
det_J = 0.25 * k * l
W = 0
for i in np.arange(0, N, 1):
    for j in np.arange(0, M, 1):
        p1 = p[i, j]
        p2 = p[i, j + 1]
        p3 = p[i + 1, j + 1]
        p4 = p[i + 1, j]

        N11, N12, N13, N14 = tinh_N(-can3_inv, -can3_inv)
        N21, N22, N23, N24 = tinh_N(can3_inv, -can3_inv)
        N31, N32, N33, N34 = tinh_N(can3_inv, can3_inv)
        N41, N42, N43, N44 = tinh_N(-can3_inv, can3_inv)

        PG1 = p1 * N11 + p2 * N12 + p3 * N13 + p4 * N14
        PG2 = p1 * N21 + p2 * N22 + p3 * N23 + p4 * N24
        PG3 = p1 * N31 + p2 * N32 + p3 * N33 + p4 * N34
        PG4 = p1 * N41 + p2 * N42 + p3 * N43 + p4 * N44

        if PG1 + PG2 + PG3 + PG4 > 0:
            W += (PG1 + PG2 + PG3 + PG4)

W = W * det_J
# => lực nâng có thứ nguyên
W_tn = W * 6 * nuy * omega * ((R / c) ** 2) * (R * L)

delta_theta = k
tan_alpha = (h_max - h_min) / np.pi

tich_phan_1 = 0
tich_phan_2 = 0

for i in np.arange(0, np.pi, 2 * np.pi / N):
    h_theta = c * (1 + epsilon * np.cos(i))
    h_theta_inv = 1 / h_theta
    tich_phan_1 += h_theta_inv * delta_theta

for i in np.arange(np.pi, 2 * np.pi + 2 * np.pi / N, 2 * np.pi / N):
    h_theta = c * (1 + epsilon * np.cos(i))
    h_theta_square = h_theta ** 2
    h_theta_square_inv = 1 / h_theta_square
    tich_phan_2 += h_theta_square_inv * delta_theta

tich_phan_2 = tich_phan_2 * h_min
tich_phan = tich_phan_1 + tich_phan_2

Ca = 0.5 * e * W_tn * np.sin(phi * np.pi / 180) + (nuy * omega * R ** 3) * L * tich_phan

# tính lưu lượng theo phương x
deltaX = X / N
delta_theta = 2 * np.pi / N

heso = 6 * nuy * omega * (R / c) ** 2
Q = []
for i in np.arange(1, N):
    dp = (P[i + 1] - P[i - 1]) / (2 * deltaX)
    hi = c * (1 + epsilon * np.cos(delta_theta * i))
    Qi = L * (0.5 * hi * v - 0.5 * omega * ((R / c) ** 2) * (hi ** 3) * dp)
    Q.append(Qi)

Q = np.asarray(Q)
q = np.mean(Q, axis=1)
plt.plot(np.arange(0, X - X / N, X / N), q, c='g')
plt.suptitle('Đồ thị lưu lượng theo phương X', fontsize=20)
plt.xlabel("Chu vi(m)", fontsize=15)
plt.ylabel("Lưu lượng (m^3/s)", fontsize=15)
plt.grid()
plt.show()
print("Khả năng tải của ổ là W = {:0.5f} N".format(W_tn))
print("Momen ma sát là : Ca = {:0.5f} N.m".format(Ca))

# Lưu lượng theo phương z
deltaX = X / N
deltaZ = L / N
deltay = h_max / N
delta_theta = 2 * np.pi / N

# Chia theo x để tìm hi tại mỗi vị trí xi
Hi = []
for i in range(N + 1):
    Hi.append(c * (1 + epsilon * np.cos(delta_theta * i)))

hs1 = 3 * omega * (R / c) ** 2
Wz = []
for i, hi in enumerate(Hi):
    n = int(hi / deltay + 1)
    Yi = np.zeros((51,))
    for j in range(n):
        yi = deltay * j
        Yi[j] = yi
    hs2 = Yi * (Yi - hi)
    if i != 0 and i < N:
        dp_dz = (P[:, i + 1] - P[:, i - 1]) / (2 * deltaZ)
        Wz.append(hs1 * dp_dz * hs2)

Wz = np.asarray(Wz)
Wz = np.concatenate((np.zeros((1, N + 1)), Wz, np.zeros((1, N + 1))), axis=0)


def tinh_N(si, ni):
    N1 = 0.25 * (1 - si) * (1 - ni)
    N2 = 0.25 * (1 + si) * (1 - ni)
    N3 = 0.25 * (1 + si) * (1 + ni)
    N4 = 0.25 * (1 - si) * (1 + ni)

    return N1, N2, N3, N4


can3_inv = 1 / np.sqrt(3)
det_J = 0.25 * deltaZ * deltay
Qz = 0
for i in np.arange(0, N, 1):
    for j in np.arange(0, M, 1):
        wz1 = Wz[i, j]
        wz2 = Wz[i, j + 1]
        wz3 = Wz[i + 1, j + 1]
        wz4 = Wz[i + 1, j]

        N11, N12, N13, N14 = tinh_N(-can3_inv, -can3_inv)
        N21, N22, N23, N24 = tinh_N(can3_inv, -can3_inv)
        N31, N32, N33, N34 = tinh_N(can3_inv, can3_inv)
        N41, N42, N43, N44 = tinh_N(-can3_inv, can3_inv)

        WG1 = wz1 * N11 + wz2 * N12 + wz3 * N13 + wz4 * N14
        WG2 = wz1 * N21 + wz2 * N22 + wz3 * N23 + wz4 * N24
        WG3 = wz1 * N31 + wz2 * N32 + wz3 * N33 + wz4 * N34
        WG4 = wz1 * N41 + wz2 * N42 + wz3 * N43 + wz4 * N44

        # if WG1 >= 0 and WG2 >= 0  and WG3 >= 0 and WG4 >= 0:
        if WG1 + WG2 + WG3 + WG4 > 0:
            Qz += (WG1 + WG2 + WG3 + WG4)

Qz = Qz * det_J
print("Lưu lượng theo phương dọc trục là : Qz = {} m^3/s".format(Qz))