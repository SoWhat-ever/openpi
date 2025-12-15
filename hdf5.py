import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt


def print_hdf5_element(hdf5_path, element, frame):
    with h5py.File(hdf5_path, "r") as f:
        print(f"frame = {frame}, {element}:", f[element][frame])


def print_hdf5_info(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        def print_h5(name, obj):
            print(name, obj)
        f.visititems(print_h5)

def show_hdf5_image(hdf5_path, camera, frame):
    with h5py.File(hdf5_path, "r") as f:
        buf = f[camera][frame]
        valid_len = int(f["compress_len"][0, frame])
        print(valid_len)
        # 截取有效字节
        jpeg_bytes = np.array(buf[:valid_len], dtype=np.uint8)
        # 解码成图像
        img = cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode 返回 None, 说明 JPEG bytes 可能损坏")
        # BGR -> RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 用 matplotlib 显示
    plt.figure(figsize=(5,5))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()


print_hdf5_element("episode_0.hdf5", "base_action", 10)
print_hdf5_info("episode_0.hdf5")
show_hdf5_image("episode_0.hdf5", "observations/images/cam_right_wrist", 900)