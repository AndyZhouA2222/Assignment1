# -*- coding: utf-8 -*-
"""
AtlasCut 图集取图  v1.0
Blender 4.0.2+ / 5.x

用途
  对着贴花图集拖一个矩形, 直接得到一张比例正确、UV 锁死的贴花平面,
  再交给「yo的贴花」接管(吸附/包裹/磨损/Bridge 全套)。

设计要点
  · 只做"生成平面 + 建材质", 不复制 yo的贴花 的任何管线。
  · UV 直接指向图集子区, 同一图集共享一份材质与一份图像数据块。
  · 遮罩不是 alpha 通道, 而是单通道软遮罩(类 SDF), 由节点组用两条
    smoothstep 重建轮廓 —— 见 Dropship_Decal 贴图处理说明。
  · 软遮罩图集的矩形要"外扩"留羽化, 硬边图集才内缩半像素防溢色。
"""

bl_info = {
    "name": "AtlasCut 图集取图",
    "author": "Annieva & Claude",
    "version": (1, 6, 1),
    "blender": (4, 0, 0),
    "location": "图像编辑器 / 3D视图 > 侧栏(N) > 图集取图",
    "description": "从贴花图集框选子区生成贴花平面, 交由 yo的贴花 接管",
    "category": "Material",
}

import os

import bpy
import blf
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader


# ============================================================
#  常量 / 契约
# ============================================================

AC_MARK    = "ac_atlas"        # 图集图像名 —— 外部平面的识别标记
AC_RECT    = "ac_rect"         # [u0, v0, u1, v1] 归一化矩形
AC_PXRECT  = "ac_px_rect"      # [x0, y0, x1, y1] 整数像素矩形
AC_CHANNEL = "ac_channel"
AC_INVERT  = "ac_invert"
AC_MAT_SIG = "ac_mat_sig"

GROUP_NAME = "AC_软遮罩解码"
GROUP_VER  = 2
NODE_IMG   = "AC_图集"
NODE_SPLIT = "AC_通道"
NODE_DEC   = "AC_解码"

# md §6: Alpha Ramp 0.445454 → 0.540910, BaseColor Ramp 0.509091 → 0.554546
ALPHA_C0, ALPHA_C1 = 0.445454, 0.540910
COLOR_C0, COLOR_C1 = 0.509091, 0.554546
MASK_CENTER = (ALPHA_C0 + ALPHA_C1) * 0.5          # 0.493182
ALPHA_WIDTH = ALPHA_C1 - ALPHA_C0                  # 0.095456
COLOR_WIDTH = COLOR_C1 - COLOR_C0                  # 0.045455
COLOR_OFFSET = (COLOR_C0 + COLOR_C1) * 0.5 - MASK_CENTER   # +0.038636
COLOR_RATIO = COLOR_WIDTH / ALPHA_WIDTH            # 0.476190

# 阈值窗口必须和源图的边缘梯度匹配, 否则会把抗锯齿吃掉:
#   软遮罩(类SDF)羽化带几十像素宽 → 窄窗口重建出锐利轮廓 (md §6)
#   硬边图集只有1像素抗锯齿 → 窄窗口会把它量化回纹理格点, 出现锯齿褶皱
SOFT_WIDTH = ALPHA_WIDTH                           # 0.0955
HARD_WIDTH = 0.85                                  # 近似直通, 保留源抗锯齿

CHANNEL_ITEMS = [
    ('R', "红 R", "红通道(Marathon 贴花图集的默认通道)"),
    ('G', "绿 G", "绿通道 —— 同一张图常打包了别的图形"),
    ('B', "蓝 B", "蓝通道"),
    ('A', "Alpha", "图像自带的透明通道"),
    ('L', "明度", "RGB 明度加权"),
]

CHANNEL_INDEX = {'R': 0, 'G': 1, 'B': 2, 'A': 3}


# ============================================================
#  像素读取与分析
# ============================================================

_PX_CACHE = {}


ANALYSIS_LIMIT = 2048       # 分析用的最长边上限; 4K 图集会被降到 1/2


def _downsample(arr, limit=ANALYSIS_LIMIT):
    """整数倍块平均降采样, 返回 (数组, 倍数)。
    分析只用于判极性与软硬, 不需要原分辨率 —— 4096² RGBA 全量是 268MB。"""
    h, w = arr.shape[0], arr.shape[1]
    k = 1
    while max(w, h) // (k * 2) >= limit:
        k *= 2
    if k == 1:
        return arr, 1
    hh, ww = (h // k) * k, (w // k) * k
    a = arr[:hh, :ww].reshape(hh // k, k, ww // k, k, arr.shape[2])
    return np.ascontiguousarray(a.mean(axis=(1, 3)).astype(np.float32)), k


def image_array(img):
    """取图像像素为 ((h, w, c) float32, 降采样倍数); 按图像名+尺寸缓存。"""
    if img is None:
        return None, 1
    w, h = img.size
    c = img.channels
    if w == 0 or h == 0 or c == 0:
        return None, 1
    key = (img.name, w, h, c)
    hit = _PX_CACHE.get(key)
    if hit is not None:
        return hit
    try:
        buf = np.empty(w * h * c, dtype=np.float32)
        img.pixels.foreach_get(buf)
    except Exception:
        return None, 1                  # 边界: 图像未加载或已失效
    arr = buf.reshape(h, w, c)          # Blender 首行在底部, 与 v 轴同向
    arr, k = _downsample(arr)
    del buf
    _PX_CACHE.clear()                   # 只留一张
    _PX_CACHE[key] = (arr, k)
    return arr, k


def channel_plane(arr, channel):
    """按通道取单通道平面。"""
    c = arr.shape[2]
    if channel == 'L':
        if c >= 3:
            return (arr[:, :, 0] * 0.2126 + arr[:, :, 1] * 0.7152
                    + arr[:, :, 2] * 0.0722)
        return arr[:, :, 0]
    i = CHANNEL_INDEX.get(channel, 0)
    if i >= c:
        i = 0
    return arr[:, :, i]


def analyze_rect(img, channel, x0, y0, x1, y1):
    """返回 (是否软遮罩, 背景是否为亮值)。读不到像素时返回 (False, False)。"""
    arr, k = image_array(img)
    if arr is None:
        return False, False
    x0 //= k; x1 //= k; y0 //= k; y1 //= k       # 换算到分析分辨率
    h, w = arr.shape[0], arr.shape[1]
    x0 = max(0, min(w - 1, x0)); x1 = max(x0 + 1, min(w, x1))
    y0 = max(0, min(h - 1, y0)); y1 = max(y0 + 1, min(h, y1))
    sub = channel_plane(arr, channel)[y0:y1, x0:x1]
    if sub.size == 0:
        return False, False
    # 软遮罩判据: 中间调占比高 —— 硬边黑白图几乎只有两端。
    # 降采样本身会在边界造出灰边, 所以带宽收窄、门槛提高。
    mid = float(np.mean((sub > 0.2) & (sub < 0.8)))
    # 背景取四边一圈; 用中位数而非均值 —— 图形压到框边时不会被带偏
    ring = np.concatenate([sub[0, :].ravel(), sub[-1, :].ravel(),
                           sub[:, 0].ravel(), sub[:, -1].ravel()])
    return mid > 0.25, float(np.median(ring)) > 0.5


# ============================================================
#  矩形规整
# ============================================================

def content_bbox(img, channel, invert, x0, y0, x1, y1, min_px):
    """在像素矩形内找出图形的实际包围框。

    图集的网格很少和图标严格对齐, 直接按格子切会得到一堆偏心、比例
    错误的贴花; 先收紧到内容才有意义。图形像素太少视为空格, 返回 None。
    """
    arr, k = image_array(img)
    if arr is None:
        return None
    h, w = arr.shape[0], arr.shape[1]
    ax0 = max(0, min(w - 1, x0 // k)); ax1 = max(ax0 + 1, min(w, x1 // k))
    ay0 = max(0, min(h - 1, y0 // k)); ay1 = max(ay0 + 1, min(h, y1 // k))
    sub = channel_plane(arr, channel)[ay0:ay1, ax0:ax1]
    if sub.size == 0:
        return None
    m = (1.0 - sub) if invert else sub       # 统一成 图形≈0 / 背景≈1
    mask = m < 0.5
    if int(mask.sum()) * k * k < max(1, min_px):
        return None
    ys, xs = np.nonzero(mask)
    bx0 = max(x0, (ax0 + int(xs.min())) * k)
    bx1 = min(x1, (ax0 + int(xs.max()) + 1) * k)
    by0 = max(y0, (ay0 + int(ys.min())) * k)
    by1 = min(y1, (ay0 + int(ys.max()) + 1) * k)
    if bx1 - bx0 < 2 or by1 - by0 < 2:
        return None
    return bx0, by0, bx1, by1


def snap_rect(img, u0, v0, u1, v1, margin_px, soft):
    """吸附到整数像素并施加边距。
    软遮罩外扩 margin_px, 硬边图额外内缩半像素防双线性吃到邻居。
    返回 ((x0, y0, x1, y1), (u0, v0, u1, v1))。"""
    w, h = img.size
    x0 = int(round(min(u0, u1) * w)); x1 = int(round(max(u0, u1) * w))
    y0 = int(round(min(v0, v1) * h)); y1 = int(round(max(v0, v1) * h))
    m = int(margin_px)
    x0 -= m; y0 -= m; x1 += m; y1 += m
    x0 = max(0, min(w - 1, x0)); x1 = max(x0 + 1, min(w, x1))
    y0 = max(0, min(h - 1, y0)); y1 = max(y0 + 1, min(h, y1))
    inset = 0.0 if soft else 0.5        # 像素单位
    fu0 = (x0 + inset) / float(w); fu1 = (x1 - inset) / float(w)
    fv0 = (y0 + inset) / float(h); fv1 = (y1 - inset) / float(h)
    if fu1 <= fu0:
        fu0, fu1 = x0 / float(w), x1 / float(w)
    if fv1 <= fv0:
        fv0, fv1 = y0 / float(h), y1 / float(h)
    return (x0, y0, x1, y1), (fu0, fv0, fu1, fv1)


# ============================================================
#  节点组
# ============================================================

def _new_socket(ng, name, in_out, stype):
    """4.0+ 走 interface, 3.x 走 inputs/outputs。"""
    if hasattr(ng, "interface"):
        return ng.interface.new_socket(name=name, in_out=in_out,
                                       socket_type=stype)
    coll = ng.inputs if in_out == 'INPUT' else ng.outputs
    return coll.new(stype, name)


def _sock_range(sock, default=None, lo=None, hi=None):
    for attr, val in (("default_value", default),
                      ("min_value", lo), ("max_value", hi)):
        if val is None:
            continue
        try:
            setattr(sock, attr, val)
        except Exception:
            pass                        # 部分 socket 类型无 min/max


def _math(nodes, op, loc, v1=None, name=""):
    n = nodes.new('ShaderNodeMath')
    n.operation = op
    n.location = loc
    if name:
        n.name = n.label = name
    if v1 is not None:
        n.inputs[1].default_value = v1
    return n


def _map_range(nodes, loc, name=""):
    n = nodes.new('ShaderNodeMapRange')
    n.location = loc
    n.clamp = True
    if hasattr(n, "interpolation_type"):
        n.interpolation_type = 'SMOOTHSTEP'
    if name:
        n.name = n.label = name
    n.inputs["To Min"].default_value = 1.0
    n.inputs["To Max"].default_value = 0.0
    return n


def ensure_decode_group():
    """建立(或复用)软遮罩解码组。

    m = |v - 反相|            —— 统一为 图形≈0 / 背景≈1
    Alpha = 1 - smoothstep(c-半宽, c+半宽, m)
    基础色 = 贴花颜色 × (1 - smoothstep(...))   两条阈值故意错开, 防黑边
    """
    ng = bpy.data.node_groups.get(GROUP_NAME)
    if ng is not None and ng.get("ac_group_version") == GROUP_VER:
        return ng
    if ng is not None:
        ng.name = GROUP_NAME + "_旧"          # 版本不符: 留旧组给旧材质
        ng = None

    ng = bpy.data.node_groups.new(GROUP_NAME, 'ShaderNodeTree')
    ng.use_fake_user = True
    ng["ac_group_version"] = GROUP_VER

    s = _new_socket(ng, "遮罩", 'INPUT', 'NodeSocketFloat')
    _sock_range(s, 0.0, 0.0, 1.0)
    s = _new_socket(ng, "反相", 'INPUT', 'NodeSocketFloat')
    _sock_range(s, 0.0, 0.0, 1.0)
    s = _new_socket(ng, "过渡宽度", 'INPUT', 'NodeSocketFloat')
    _sock_range(s, ALPHA_WIDTH, 0.002, 0.5)
    s = _new_socket(ng, "轮廓膨胀", 'INPUT', 'NodeSocketFloat')
    _sock_range(s, 0.0, -0.3, 0.3)
    s = _new_socket(ng, "贴花颜色", 'INPUT', 'NodeSocketColor')
    _sock_range(s, (1.0, 1.0, 1.0, 1.0))
    s = _new_socket(ng, "色边跟随", 'INPUT', 'NodeSocketFloat')
    _sock_range(s, 1.0, 0.0, 1.0)

    _new_socket(ng, "基础色", 'OUTPUT', 'NodeSocketColor')
    _new_socket(ng, "Alpha", 'OUTPUT', 'NodeSocketFloat')
    _new_socket(ng, "遮罩m", 'OUTPUT', 'NodeSocketFloat')

    nodes, links = ng.nodes, ng.links
    gi = nodes.new('NodeGroupInput');  gi.location = (-1100, 0)
    go = nodes.new('NodeGroupOutput'); go.location = (700, 0)

    sub = _math(nodes, 'SUBTRACT', (-880, 160), name="极性")
    links.new(gi.outputs["遮罩"], sub.inputs[0])
    links.new(gi.outputs["反相"], sub.inputs[1])
    m = _math(nodes, 'ABSOLUTE', (-700, 160), name="遮罩m")
    links.new(sub.outputs[0], m.inputs[0])

    half = _math(nodes, 'MULTIPLY', (-880, -80), v1=0.5, name="半宽")
    links.new(gi.outputs["过渡宽度"], half.inputs[0])
    center = _math(nodes, 'ADD', (-880, -260), v1=MASK_CENTER, name="中心")
    links.new(gi.outputs["轮廓膨胀"], center.inputs[0])

    a0 = _math(nodes, 'SUBTRACT', (-520, -120), name="A下限")
    links.new(center.outputs[0], a0.inputs[0])
    links.new(half.outputs[0], a0.inputs[1])
    a1 = _math(nodes, 'ADD', (-520, -300), name="A上限")
    links.new(center.outputs[0], a1.inputs[0])
    links.new(half.outputs[0], a1.inputs[1])

    alpha = _map_range(nodes, (-200, 120), name="Alpha轮廓")
    links.new(m.outputs[0], alpha.inputs["Value"])
    links.new(a0.outputs[0], alpha.inputs["From Min"])
    links.new(a1.outputs[0], alpha.inputs["From Max"])

    bhalf = _math(nodes, 'MULTIPLY', (-700, -480), v1=COLOR_RATIO,
                  name="色半宽")
    links.new(half.outputs[0], bhalf.inputs[0])
    bcen = _math(nodes, 'ADD', (-700, -660), v1=COLOR_OFFSET, name="色中心")
    links.new(center.outputs[0], bcen.inputs[0])
    b0 = _math(nodes, 'SUBTRACT', (-520, -520), name="色下限")
    links.new(bcen.outputs[0], b0.inputs[0])
    links.new(bhalf.outputs[0], b0.inputs[1])
    b1 = _math(nodes, 'ADD', (-520, -700), name="色上限")
    links.new(bcen.outputs[0], b1.inputs[0])
    links.new(bhalf.outputs[0], b1.inputs[1])

    white = _map_range(nodes, (-200, -400), name="色轮廓")
    links.new(m.outputs[0], white.inputs["Value"])
    links.new(b0.outputs[0], white.inputs["From Min"])
    links.new(b1.outputs[0], white.inputs["From Max"])

    # 色边跟随=0 时基础色恒等于贴花颜色, 只靠 alpha 抗锯齿。
    # 硬边图集必须这样: 宽阈值下色带会把边缘像素压暗, 出现黑边。
    dark = _math(nodes, 'SUBTRACT', (0, -400), name="色带强度")
    dark.inputs[0].default_value = 1.0
    links.new(white.outputs["Result"], dark.inputs[1])
    scaled = _math(nodes, 'MULTIPLY', (150, -400), name="跟随量")
    links.new(gi.outputs["色边跟随"], scaled.inputs[0])
    links.new(dark.outputs[0], scaled.inputs[1])
    white_eff = _math(nodes, 'SUBTRACT', (300, -400), name="有效白度")
    white_eff.inputs[0].default_value = 1.0
    links.new(scaled.outputs[0], white_eff.inputs[1])

    # 颜色 × 白度: 用 VectorMath 而非 MixRGB —— 跨 4.x/5.x 都稳定
    tint = nodes.new('ShaderNodeVectorMath')
    tint.operation = 'MULTIPLY'
    tint.location = (480, -180)
    tint.name = tint.label = "着色"
    links.new(gi.outputs["贴花颜色"], tint.inputs[0])
    links.new(white_eff.outputs[0], tint.inputs[1])

    links.new(tint.outputs["Vector"], go.inputs["基础色"])
    links.new(alpha.outputs["Result"], go.inputs["Alpha"])
    links.new(m.outputs[0], go.inputs["遮罩m"])

    # 版本升级后把已有材质挂到新组上。输入是超集且名称不变,
    # 已连好的线按 socket 标识保留, 新输入取默认值。
    for mat in bpy.data.materials:
        if AC_MARK not in mat or not mat.use_nodes:
            continue
        node = mat.node_tree.nodes.get(NODE_DEC)
        if (node is not None and node.node_tree is not None
                and node.node_tree is not ng
                and node.node_tree.name.startswith(GROUP_NAME)):
            node.node_tree = ng
    return ng


# ============================================================
#  烘焙: 子区 → 独立 RGBA 小图
# ============================================================

def _smoothstep(a, b, x):
    d = b - a
    if abs(d) < 1e-9:
        d = 1e-9
    t = np.clip((x - a) / d, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def full_pixels(img):
    """全分辨率读取(不进缓存)。烘焙必须用原分辨率, 不能用分析用的降采样。"""
    w, h = img.size
    c = img.channels
    if w == 0 or h == 0 or c == 0:
        return None
    try:
        buf = np.empty(w * h * c, dtype=np.float32)
        img.pixels.foreach_get(buf)
    except Exception:
        return None
    return buf.reshape(h, w, c)


def bake_cut(arr, px, channel, invert, edge_width, dilate, color, follow,
             name, out_path):
    """把子区按解码组的同一套数学烘成独立 PNG。

    RGB = 贴花颜色 × 有效白度, A = 解码后的 alpha。烘出来的图自带
    alpha, 之后走 yo的贴花 的普通导入路径即可, 不再依赖图集与解码组。
    """
    x0, y0, x1, y1 = px
    h, w = arr.shape[0], arr.shape[1]
    x0 = max(0, min(w - 1, x0)); x1 = max(x0 + 1, min(w, x1))
    y0 = max(0, min(h - 1, y0)); y1 = max(y0 + 1, min(h, y1))
    sub = channel_plane(arr, channel)[y0:y1, x0:x1]
    if sub.size == 0:
        return None, "子区为空"
    m = np.abs(sub - (1.0 if invert else 0.0))

    half = edge_width * 0.5
    center = MASK_CENTER + dilate
    alpha = 1.0 - _smoothstep(center - half, center + half, m)
    bhalf = half * COLOR_RATIO
    bcen = center + COLOR_OFFSET
    white = 1.0 - _smoothstep(bcen - bhalf, bcen + bhalf, m)
    eff = 1.0 - follow * (1.0 - white)

    bh, bw = sub.shape
    out = np.empty((bh, bw, 4), dtype=np.float32)
    for i in range(3):
        out[:, :, i] = color[i] * eff
    out[:, :, 3] = alpha

    image = bpy.data.images.new(name, width=bw, height=bh, alpha=True)
    try:
        image.pixels.foreach_set(out.ravel())
        image.filepath_raw = out_path
        image.file_format = 'PNG'
        image.save()
    except Exception as exc:
        try:
            bpy.data.images.remove(image)
        except Exception:
            pass
        return None, str(exc)
    try:
        bpy.data.images.remove(image)   # 磁盘上的文件才是产物, 数据块不留
    except Exception:
        pass
    return out_path, "%d×%d" % (bw, bh)


# ============================================================
#  材质
# ============================================================

def material_signature(img, channel, invert, blend):
    return "AC|%s|%s|%d|%s" % (img.name, channel, 1 if invert else 0, blend)


def material_is_customized(mat):
    """已被 yo的贴花 接手(Bridge / 磨损 / 匹配 / 浮雕 / 分家)的材质不再共享。"""
    if mat is None or not mat.use_nodes:
        return True
    if mat.get("sd_no_merge") or mat.get("sd_bridge_version"):
        return True
    for n in mat.node_tree.nodes:
        if n.name.startswith(("SD_", "SDS_", "SDR_", "SDB_", "SDW_")):
            return True
        if (n.bl_idname == 'ShaderNodeGroup' and n.node_tree is not None
                and n.node_tree.name.startswith("SD_")):
            return True
    return False


def find_shared_material(sig):
    for m in bpy.data.materials:
        if (m.get(AC_MAT_SIG) == sig and m.use_nodes
                and not material_is_customized(m)):
            return m
    return None


def apply_blend(mat, blend):
    """4.0/4.1 旧 EEVEE 与 4.2+ EEVEE Next 双写。"""
    if hasattr(mat, "blend_method"):
        mat.blend_method = blend
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = 'NONE'
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = ('BLENDED' if blend == 'BLEND'
                                     else 'DITHERED')


def wire_channel(nt, channel):
    """把图集节点的指定通道接到解码组的「遮罩」输入。
    会先拆掉旧的分离/明度节点, 所以可以反复调用来换通道。"""
    tex = nt.nodes.get(NODE_IMG)
    dec = nt.nodes.get(NODE_DEC)
    if tex is None or dec is None:
        return False
    dst = dec.inputs.get("遮罩")
    if dst is None:
        return False
    for link in list(dst.links):
        nt.links.remove(link)
    old = nt.nodes.get(NODE_SPLIT)
    if old is not None:
        nt.nodes.remove(old)
    if channel == 'A':
        nt.links.new(tex.outputs["Alpha"], dst)
        return True
    if channel == 'L':
        bw = nt.nodes.new('ShaderNodeRGBToBW')
        bw.name = bw.label = NODE_SPLIT
        bw.location = (tex.location[0] + 300.0, tex.location[1] + 40.0)
        nt.links.new(tex.outputs["Color"], bw.inputs["Color"])
        nt.links.new(bw.outputs["Val"], dst)
        return True
    sep = nt.nodes.new('ShaderNodeSeparateColor')
    sep.name = sep.label = NODE_SPLIT
    sep.location = (tex.location[0] + 300.0, tex.location[1] + 40.0)
    if hasattr(sep, "mode"):
        sep.mode = 'RGB'
    nt.links.new(tex.outputs["Color"], sep.inputs["Color"])
    nt.links.new(sep.outputs[{'R': "Red", 'G': "Green",
                              'B': "Blue"}[channel]], dst)
    return True


def build_material(img, p, invert):
    """建一份图集材质: 图像 → 通道 → 解码组 → Principled。"""
    name = "atlas_%s_%s" % (os.path.splitext(img.name)[0], p.channel)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        if n.bl_idname not in {'ShaderNodeBsdfPrincipled',
                               'ShaderNodeOutputMaterial'}:
            nt.nodes.remove(n)
    bsdf = next((n for n in nt.nodes
                 if n.bl_idname == 'ShaderNodeBsdfPrincipled'), None)
    if bsdf is None:
        bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.location = (300, 0)
    out = next((n for n in nt.nodes
                if n.bl_idname == 'ShaderNodeOutputMaterial'), None)
    if out is None:
        out = nt.nodes.new('ShaderNodeOutputMaterial')
        out.location = (620, 0)
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])

    tex = nt.nodes.new('ShaderNodeTexImage')
    tex.name = tex.label = NODE_IMG
    tex.image = img
    tex.location = (-820, 0)
    tex.interpolation = p.interp
    tex.extension = 'EXTEND'
    try:
        img.colorspace_settings.name = 'Non-Color'   # md §1: 避免 sRGB 改阈值
    except Exception:
        pass

    dec = nt.nodes.new('ShaderNodeGroup')
    dec.node_tree = ensure_decode_group()
    dec.name = dec.label = NODE_DEC
    dec.location = (-260, 0)

    wire_channel(nt, p.channel)

    dec.inputs["反相"].default_value = 1.0 if invert else 0.0
    dec.inputs["过渡宽度"].default_value = p.edge_width
    dec.inputs["轮廓膨胀"].default_value = p.dilate
    dec.inputs["贴花颜色"].default_value = (p.color[0], p.color[1],
                                            p.color[2], 1.0)
    if "色边跟随" in dec.inputs:
        dec.inputs["色边跟随"].default_value = p.color_follow

    nt.links.new(dec.outputs["基础色"], bsdf.inputs["Base Color"])
    nt.links.new(dec.outputs["Alpha"], bsdf.inputs["Alpha"])
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.62
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = 0.0

    apply_blend(mat, p.blend)
    mat[AC_MAT_SIG] = material_signature(img, p.channel, invert, p.blend)
    mat[AC_MARK] = img.name
    return mat


def get_material(img, p, invert):
    if p.share_material:
        sig = material_signature(img, p.channel, invert, p.blend)
        m = find_shared_material(sig)
        if m is not None:
            return m
    return build_material(img, p, invert)


# ============================================================
#  网格
# ============================================================

def build_plane_mesh(name, sx, sy, div, rect):
    """带图集 UV 的细分平面, 面朝 +Z, 原点居中。"""
    u0, v0, u1, v1 = rect
    div = max(1, int(div))
    n = div + 1
    verts = []
    for j in range(n):
        fy = j / float(div)
        for i in range(n):
            fx = i / float(div)
            verts.append(((fx - 0.5) * sx, (fy - 0.5) * sy, 0.0))
    faces = []
    for j in range(div):
        for i in range(div):
            a = j * n + i
            faces.append((a, a + 1, a + n + 1, a + n))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    uv = me.uv_layers.new(name="UVMap")
    ix = 1.0 / sx if sx else 1.0
    iy = 1.0 / sy if sy else 1.0
    for loop in me.loops:
        co = me.vertices[loop.vertex_index].co
        fx = co.x * ix + 0.5
        fy = co.y * iy + 0.5
        uv.data[loop.index].uv = (u0 + fx * (u1 - u0), v0 + fy * (v1 - v0))
    return me


def retarget_uv(obj, rect):
    """把已有贴花的 UV 重新指到新矩形(按原 UV 的相对位置重映射)。"""
    me = obj.data
    if not me.uv_layers:
        return False
    old = obj.get(AC_RECT)
    if not old or len(old) != 4:
        return False
    ou0, ov0, ou1, ov1 = [float(x) for x in old]
    du = (ou1 - ou0) or 1.0
    dv = (ov1 - ov0) or 1.0
    u0, v0, u1, v1 = rect
    uv = me.uv_layers.active or me.uv_layers[0]
    for d in uv.data:
        fx = (d.uv[0] - ou0) / du
        fy = (d.uv[1] - ov0) / dv
        d.uv = (u0 + fx * (u1 - u0), v0 + fy * (v1 - v0))
    me.update()
    return True


# ============================================================
#  尺寸
# ============================================================

def is_yo_decal(obj):
    """粗判是否已是 yo的贴花 管理的贴花(不做目标)。"""
    if obj is None or obj.type != 'MESH':
        return False
    if ("sd_decal_id" in obj or "surfdecal_target" in obj
            or "surfdecal_rest_mesh" in obj or AC_MARK in obj):
        return True
    return any(m.name.startswith("SD_") for m in obj.modifiers)


def pick_target(context):
    """尺寸参照物: 活动物体优先, 其次任一所选; 排除贴花本身。"""
    ao = context.active_object
    cands = ([ao] if ao is not None else []) + list(context.selected_objects)
    for o in cands:
        if o is None or is_yo_decal(o):
            continue
        if o.type == 'MESH':
            return o
        if o.type == 'EMPTY' and o.instance_collection is not None:
            return o
    return None


def compute_size(context, p, px_w, px_h, target=None):
    """返回 (sx, sy, 说明)。比例始终按像素矩形锁死, 只决定最长边多长。

    target 可由调用方指定: 批量切分时每生成一张选择就被改写了,
    再去 pick_target 只会找到刚生成的贴花。
    """
    px_w = max(1, px_w); px_h = max(1, px_h)
    aspect = px_w / float(px_h)
    if p.size_mode != 'TARGET':
        target = None
    elif target is None:
        target = pick_target(context)
    if target is not None:
        dims = target.dimensions
        base = max(dims[0], dims[1], dims[2])
        if base <= 0.0:
            base = 1.0
        longest = base * p.target_pct
        note = "%s 包围盒 %.2fm × %d%%" % (target.name, base,
                                           round(p.target_pct * 100))
    else:
        longest = max(px_w, px_h) / max(1e-6, p.px_per_m)
        note = ("%.0f px/m" % p.px_per_m) if p.size_mode == 'DENSITY' \
            else "无目标, 回退 %.0f px/m" % p.px_per_m
    longest = max(longest, p.min_size)      # 下限, 避免生成一堆迷你贴花
    longest = max(1e-4, longest)
    if aspect >= 1.0:
        sx, sy = longest, longest / aspect
    else:
        sx, sy = longest * aspect, longest
    return sx, sy, note


# ============================================================
#  生成
# ============================================================

def create_from_rect(context, u0, v0, u1, v1, target=None):
    """核心: 矩形 → 贴花平面。返回 (obj, 信息) 或 (None, 错误)。"""
    p = context.scene.atlascut
    img = p.atlas
    if img is None:
        return None, "没有指定图集"
    w, h = img.size
    if w == 0 or h == 0:
        return None, "图集尺寸无效(未加载?)"
    if abs(u1 - u0) * w < 2.0 or abs(v1 - v0) * h < 2.0:
        return None, "矩形太小(不足2像素)"

    # 先按未加边距的矩形分析, 再决定边距方向
    px0 = int(round(min(u0, u1) * w)); px1 = int(round(max(u0, u1) * w))
    py0 = int(round(min(v0, v1) * h)); py1 = int(round(max(v0, v1) * h))
    soft, bg_bright = analyze_rect(img, p.channel, px0, py0, px1, py1)

    if p.margin_mode == 'AUTO':
        margin = p.margin_px if soft else 0
    else:
        margin = p.margin_px
    px, uv = snap_rect(img, u0, v0, u1, v1, margin, soft)

    # 阈值窗口跟着源图的边缘梯度走。硬边图集用窄窗口会把源图那 1 像素
    # 的抗锯齿量化掉, 轮廓被压回纹理格点, 表现为锯齿与褶皱。
    if p.auto_edge:
        want_w = SOFT_WIDTH if soft else HARD_WIDTH
        want_f = 1.0 if soft else 0.0
        if abs(p.edge_width - want_w) > 1e-6:
            p.edge_width = want_w       # 回调会一并同步已生成的贴花
        if abs(p.color_follow - want_f) > 1e-6:
            p.color_follow = want_f

    # 目标格式: 图形 m≈0 / 背景 m≈1。
    # 背景本就是亮的(红底低值图形) → 直接用; 背景暗(黑底白图) → 必须反相。
    if p.invert_mode == 'AUTO':
        invert = not bg_bright
    else:
        invert = (p.invert_mode == 'ON')

    px_w = px[2] - px[0]
    px_h = px[3] - px[1]
    sx, sy, note = compute_size(context, p, px_w, px_h, target)

    base = os.path.splitext(img.name)[0]
    name = "%s_%s_%04d_%04d" % (base, p.channel, px[0], px[1])
    me = build_plane_mesh(name, sx, sy, p.grid, uv)
    obj = bpy.data.objects.new(name, me)
    context.collection.objects.link(obj)
    obj.location = context.scene.cursor.location

    mat = get_material(img, p, invert)
    obj.data.materials.append(mat)

    obj[AC_MARK] = img.name
    obj[AC_RECT] = list(uv)
    obj[AC_PXRECT] = list(px)
    obj[AC_CHANNEL] = p.channel
    obj[AC_INVERT] = 1 if invert else 0

    for o in context.selected_objects:
        try:
            o.select_set(False)
        except RuntimeError:
            pass                        # 不在当前视图层
    try:
        obj.select_set(True)
        context.view_layer.objects.active = obj
    except RuntimeError:
        pass

    density = max(px_w, px_h) / max(1e-6, max(sx, sy))
    info = "%d×%d px · %.3f×%.3fm · %.0f px/m · %s · %s%s" % (
        px_w, px_h, sx, sy, density, note,
        "软遮罩外扩%dpx" % margin if soft else "硬边内缩0.5px",
        " · 已反相" if invert else "")
    return obj, info


def try_adopt(context, obj, report):
    """交给 yo的贴花 接管; 未安装则只提示。"""
    p = context.scene.atlascut
    if p.after != 'ADOPT':
        return
    op = getattr(bpy.ops, "surfdecal", None)
    if op is None or not hasattr(op, "adopt_external"):
        report({'WARNING'}, "未找到 yo的贴花 的[接管外部平面], 平面已生成")
        return
    try:
        op.adopt_external('EXEC_DEFAULT', after=p.adopt_after)
    except Exception as e:              # 边界: 跨插件调用失败不该丢掉平面
        report({'WARNING'}, "接管失败(%s), 平面已生成" % e)


# ============================================================
#  操作符
# ============================================================

class ATLASCUT_OT_load(bpy.types.Operator):
    """打开图集图像文件"""
    bl_idname = "atlascut.load"
    bl_label = "打开图集"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    filter_glob: bpy.props.StringProperty(
        default="*.png;*.jpg;*.jpeg;*.tga;*.tif;*.tiff;*.exr",
        options={'HIDDEN'})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        try:
            img = bpy.data.images.load(self.filepath, check_existing=True)
        except Exception as e:
            self.report({'ERROR'}, "加载失败: %s" % e)
            return {'CANCELLED'}
        try:
            img.colorspace_settings.name = 'Non-Color'
        except Exception:
            pass
        context.scene.atlascut.atlas = img
        _PX_CACHE.clear()
        self.report({'INFO'}, "图集: %s (%d×%d)"
                    % (img.name, img.size[0], img.size[1]))
        return {'FINISHED'}


class ATLASCUT_OT_use_editor_image(bpy.types.Operator):
    """把当前图像编辑器里显示的图当作图集"""
    bl_idname = "atlascut.use_editor_image"
    bl_label = "用当前图"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        sp = context.space_data
        return (sp is not None and sp.type == 'IMAGE_EDITOR'
                and sp.image is not None)

    def execute(self, context):
        img = context.space_data.image
        context.scene.atlascut.atlas = img
        _PX_CACHE.clear()
        self.report({'INFO'}, "图集: %s" % img.name)
        return {'FINISHED'}


_draw_handle = None
_shader = None


def _get_shader():
    global _shader
    if _shader is None:
        try:
            _shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        except Exception:
            _shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
    return _shader


def _rect_lines(region, u0, v0, u1, v1):
    v2d = region.view2d
    pts = []
    for uu, vv in ((u0, v0), (u1, v0), (u1, v1), (u0, v1), (u0, v0)):
        pts.append(v2d.view_to_region(uu, vv, clip=False))
    return pts


def _draw_pick(op, context):
    if op.start is None or op.cur is None:
        return
    region = context.region
    if region is None:
        return
    u0, v0 = op.start
    u1, v1 = op.cur
    shader = _get_shader()
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(1.0)
    batch = batch_for_shader(shader, 'LINE_STRIP',
                             {"pos": _rect_lines(region, u0, v0, u1, v1)})
    shader.bind()
    shader.uniform_float("color", (1.0, 1.0, 1.0, 0.55))
    batch.draw(shader)

    img = context.scene.atlascut.atlas
    if img is not None and img.size[0] and img.size[1]:
        p = context.scene.atlascut
        soft = bool(op.soft)
        margin = (p.margin_px if soft else 0) if p.margin_mode == 'AUTO' \
            else p.margin_px
        px, uv = snap_rect(img, u0, v0, u1, v1, margin, soft)
        batch = batch_for_shader(shader, 'LINE_STRIP',
                                 {"pos": _rect_lines(region, *uv)})
        shader.uniform_float("color", (0.15, 0.95, 0.75, 0.95))
        batch.draw(shader)
        pos = region.view2d.view_to_region(min(uv[0], uv[2]),
                                           max(uv[1], uv[3]), clip=False)
        try:
            blf.size(0, 12)
        except TypeError:
            blf.size(0, 12, 72)         # 3.x 三参数签名
        blf.color(0, 0.15, 0.95, 0.75, 1.0)
        blf.position(0, pos[0], pos[1] + 8, 0)
        blf.draw(0, "%d × %d px%s" % (px[2] - px[0], px[3] - px[1],
                                      "  软遮罩" if soft else ""))
    gpu.state.blend_set('NONE')


class ATLASCUT_OT_pick_rect(bpy.types.Operator):
    """在图像编辑器里拖一个矩形, 松手生成贴花平面"""
    bl_idname = "atlascut.pick_rect"
    bl_label = "拖框取图"
    bl_options = {'REGISTER', 'UNDO', 'BLOCKING'}

    @classmethod
    def poll(cls, context):
        sp = context.space_data
        return (sp is not None and sp.type == 'IMAGE_EDITOR'
                and sp.image is not None)

    def _view(self, context, event):
        region = context.region
        if region is None:
            return None
        return region.view2d.region_to_view(event.mouse_region_x,
                                            event.mouse_region_y)

    def invoke(self, context, event):
        global _draw_handle
        p = context.scene.atlascut
        if p.atlas is None:
            p.atlas = context.space_data.image
        if p.atlas is None:
            self.report({'ERROR'}, "没有图集")
            return {'CANCELLED'}
        self.start = None
        self.cur = None
        self.soft = False
        self.dragging = False
        if _draw_handle is None:
            _draw_handle = bpy.types.SpaceImageEditor.draw_handler_add(
                _draw_pick, (self, context), 'WINDOW', 'POST_PIXEL')
        context.window_manager.modal_handler_add(self)
        context.area.header_text_set("拖框取图: 按住左键拖出矩形, 右键/ESC 取消")
        return {'RUNNING_MODAL'}

    def _cleanup(self, context):
        global _draw_handle
        if _draw_handle is not None:
            try:
                bpy.types.SpaceImageEditor.draw_handler_remove(
                    _draw_handle, 'WINDOW')
            except Exception:
                pass
            _draw_handle = None
        try:
            context.area.header_text_set(None)
        except Exception:
            pass
        if context.area is not None:
            context.area.tag_redraw()

    def modal(self, context, event):
        if event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
            self._cleanup(context)
            return {'CANCELLED'}

        if event.type == 'MOUSEMOVE':
            if self.dragging:
                self.cur = self._view(context, event)
                if context.area is not None:
                    context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self.start = self._view(context, event)
                self.cur = self.start
                self.dragging = True
                p = context.scene.atlascut
                img = p.atlas
                if img is not None and img.size[0]:
                    # 先按整图估一次软硬, 拖动中给出正确的边距预览
                    self.soft, _ = analyze_rect(img, p.channel, 0, 0,
                                                img.size[0], img.size[1])
                return {'RUNNING_MODAL'}
            if event.value == 'RELEASE' and self.dragging:
                self.cur = self._view(context, event)
                self._cleanup(context)
                if self.start is None or self.cur is None:
                    return {'CANCELLED'}
                obj, info = create_from_rect(context, self.start[0],
                                             self.start[1], self.cur[0],
                                             self.cur[1])
                if obj is None:
                    self.report({'WARNING'}, info)
                    return {'CANCELLED'}
                try_adopt(context, obj, self.report)
                self.report({'INFO'}, "%s: %s" % (obj.name, info))
                return {'FINISHED'}

        return {'RUNNING_MODAL'}


class ATLASCUT_OT_cut_manual(bpy.types.Operator):
    """按面板上的像素坐标取图"""
    bl_idname = "atlascut.cut_manual"
    bl_label = "按像素矩形取图"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        p = context.scene.atlascut
        img = p.atlas
        if img is None or img.size[0] == 0:
            self.report({'ERROR'}, "没有有效图集")
            return {'CANCELLED'}
        w, h = img.size
        u0 = p.px_x / float(w)
        v0 = p.px_y / float(h)
        u1 = (p.px_x + p.px_w) / float(w)
        v1 = (p.px_y + p.px_h) / float(h)
        obj, info = create_from_rect(context, u0, v0, u1, v1)
        if obj is None:
            self.report({'ERROR'}, info)
            return {'CANCELLED'}
        try_adopt(context, obj, self.report)
        self.report({'INFO'}, "%s: %s" % (obj.name, info))
        return {'FINISHED'}


class ATLASCUT_OT_cut_from_uv(bpy.types.Operator):
    """用编辑模式下选中的 UV 的包围框取图"""
    bl_idname = "atlascut.cut_from_uv"
    bl_label = "从UV选区取图"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return (o is not None and o.type == 'MESH'
                and context.mode == 'EDIT_MESH')

    def execute(self, context):
        import bmesh
        obj = context.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        layer = bm.loops.layers.uv.active
        if layer is None:
            self.report({'ERROR'}, "网格没有UV层")
            return {'CANCELLED'}
        us, vs = [], []
        for f in bm.faces:
            for lp in f.loops:
                d = lp[layer]
                if getattr(d, "select", False) or f.select:
                    us.append(d.uv[0]); vs.append(d.uv[1])
        if len(us) < 2:
            self.report({'ERROR'}, "没有选中的UV")
            return {'CANCELLED'}
        obj2, info = create_from_rect(context, min(us), min(vs),
                                      max(us), max(vs))
        if obj2 is None:
            self.report({'ERROR'}, info)
            return {'CANCELLED'}
        self.report({'INFO'}, "%s: %s" % (obj2.name, info))
        return {'FINISHED'}


class ATLASCUT_OT_cut_grid(bpy.types.Operator):
    """把图集按 列×行 一次性切成多张贴花"""
    bl_idname = "atlascut.cut_grid"
    bl_label = "网格批量切分"
    bl_options = {'REGISTER', 'UNDO'}

    cols: bpy.props.IntProperty(name="列", default=8, min=1, max=64)
    rows: bpy.props.IntProperty(name="行", default=8, min=1, max=64)
    region: bpy.props.EnumProperty(
        name="范围",
        items=[('FULL', "整张图集", "把整张图切成网格"),
               ('RECT', "面板矩形", "只切面板上那个像素矩形")],
        default='FULL')
    trim: bpy.props.BoolProperty(
        name="裁到内容", default=True,
        description="把每格收紧到图形的实际包围框; 关掉则严格按格子切")
    min_px: bpy.props.IntProperty(
        name="空格阈值(px)", default=64, min=1,
        description="格内图形像素少于此值就跳过, 用来滤掉图集的空白格")
    spacing: bpy.props.FloatProperty(
        name="摆放间距", default=1.25, min=1.0, max=4.0,
        description="生成后在场景里按网格摊开, 相对贴花尺寸的倍数")

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        p = context.scene.atlascut
        img = p.atlas
        if img is None or img.size[0] == 0:
            self.report({'ERROR'}, "没有有效图集")
            return {'CANCELLED'}
        w, h = img.size
        if self.region == 'RECT':
            rx0 = max(0, min(w - 2, p.px_x)); ry0 = max(0, min(h - 2, p.px_y))
            rx1 = min(w, rx0 + max(2, p.px_w)); ry1 = min(h, ry0 + max(2, p.px_h))
        else:
            rx0, ry0, rx1, ry1 = 0, 0, w, h
        rw = rx1 - rx0; rh = ry1 - ry0
        if rw < self.cols * 2 or rh < self.rows * 2:
            self.report({'ERROR'}, "格子小于2像素, 减少行列数")
            return {'CANCELLED'}

        # 极性先定下来: 每格单独判会因为格内几乎全是图形而判反
        if p.invert_mode == 'AUTO':
            _soft, bright = analyze_rect(img, p.channel, rx0, ry0, rx1, ry1)
            invert = not bright
        else:
            invert = (p.invert_mode == 'ON')
        target = pick_target(context)       # 只找一次

        made, skipped = [], 0
        for r in range(self.rows):
            for c in range(self.cols):
                x0 = rx0 + (rw * c) // self.cols
                x1 = rx0 + (rw * (c + 1)) // self.cols
                y1 = ry1 - (rh * r) // self.rows        # 从图顶往下排
                y0 = ry1 - (rh * (r + 1)) // self.rows
                if self.trim:
                    box = content_bbox(img, p.channel, invert,
                                       x0, y0, x1, y1, self.min_px)
                    if box is None:
                        skipped += 1
                        continue
                    x0, y0, x1, y1 = box
                obj, _info = create_from_rect(context, x0 / float(w),
                                              y0 / float(h), x1 / float(w),
                                              y1 / float(h), target)
                if obj is None:
                    skipped += 1
                    continue
                made.append((r, c, obj))

        if not made:
            self.report({'WARNING'}, "没切出任何贴花(全被当成空格?)")
            return {'CANCELLED'}

        # 按网格摊开, 否则全部叠在3D游标上
        pitch = max(max(o.dimensions[0], o.dimensions[1])
                    for _r, _c, o in made) * self.spacing
        base = context.scene.cursor.location
        for r, c, obj in made:
            obj.location = (base[0] + c * pitch, base[1] - r * pitch, base[2])
        for o in context.selected_objects:
            try:
                o.select_set(False)
            except RuntimeError:
                pass
        for _r, _c, obj in made:
            try:
                obj.select_set(True)
            except RuntimeError:
                pass
        try:
            context.view_layer.objects.active = made[0][2]
        except (RuntimeError, ReferenceError):
            pass
        self.report({'INFO'}, "切出 %d 张, 跳过 %d 个空格"
                    % (len(made), skipped))
        return {'FINISHED'}


class ATLASCUT_OT_reframe(bpy.types.Operator):
    """用面板上的像素矩形重框所选贴花: UV 与长宽比一起更新"""
    bl_idname = "atlascut.reframe"
    bl_label = "回到图集重框"
    bl_options = {'REGISTER', 'UNDO'}

    keep_size: bpy.props.BoolProperty(
        name="保持当前尺寸", default=False,
        description="只改UV与长宽比, 不重新计算物理大小")

    def execute(self, context):
        p = context.scene.atlascut
        img = p.atlas
        if img is None or img.size[0] == 0:
            self.report({'ERROR'}, "没有有效图集")
            return {'CANCELLED'}
        w, h = img.size
        u0 = p.px_x / float(w); v0 = p.px_y / float(h)
        u1 = (p.px_x + p.px_w) / float(w); v1 = (p.px_y + p.px_h) / float(h)
        soft, _ = analyze_rect(img, p.channel, p.px_x, p.px_y,
                               p.px_x + p.px_w, p.px_y + p.px_h)
        margin = (p.margin_px if soft else 0) if p.margin_mode == 'AUTO' \
            else p.margin_px
        px, uv = snap_rect(img, u0, v0, u1, v1, margin, soft)
        done = 0
        for o in context.selected_objects:
            if o.type != 'MESH' or AC_MARK not in o:
                continue
            if not retarget_uv(o, uv):
                continue
            o[AC_RECT] = list(uv)
            o[AC_PXRECT] = list(px)
            if not self.keep_size:
                sx, sy, _n = compute_size(context, p, px[2] - px[0],
                                          px[3] - px[1])
                d = o.dimensions
                if d[0] > 0 and d[1] > 0:
                    o.scale = (o.scale[0] * sx / d[0],
                               o.scale[1] * sy / d[1], o.scale[2])
            done += 1
        if not done:
            self.report({'WARNING'}, "所选里没有本插件生成的贴花平面")
            return {'CANCELLED'}
        self.report({'INFO'}, "已重框 %d 张" % done)
        return {'FINISHED'}


class ATLASCUT_OT_bake_library(bpy.types.Operator):
    """把所选贴花的图集子区烘成独立 PNG, 并收进 yo的贴花 的贴花库"""
    bl_idname = "atlascut.bake_library"
    bl_label = "烘焙并收进贴花库"
    bl_options = {'REGISTER'}

    to_library: bpy.props.BoolProperty(
        name="收进贴花库", default=True,
        description="关掉则只烘出 PNG 文件, 不登记")

    def execute(self, context):
        objs = [o for o in context.selected_objects
                if o.type == 'MESH' and AC_PXRECT in o and AC_MARK in o]
        if not objs:
            self.report({'ERROR'}, "请先选中本插件切出来的贴花")
            return {'CANCELLED'}
        p = context.scene.atlascut
        outdir = bpy.app.tempdir or os.path.dirname(bpy.data.filepath) or "."
        can_register = (self.to_library
                        and hasattr(getattr(bpy.ops, "surfdecal", None),
                                    "library_add"))
        cache = {}
        reason = []
        done, failed = [], 0
        for obj in objs:
            img = bpy.data.images.get(obj.get(AC_MARK, ""))
            if img is None:
                failed += 1
                continue
            arr = cache.get(img.name)
            if arr is None:
                arr = full_pixels(img)      # 每张图集只读一次
                if arr is None:
                    failed += 1
                    continue
                cache[img.name] = arr
            px = [int(v) for v in obj.get(AC_PXRECT, (0, 0, 2, 2))]
            out_path = os.path.join(outdir, "%s.png" % obj.name)
            got, note = bake_cut(
                arr, px, obj.get(AC_CHANNEL, p.channel),
                bool(obj.get(AC_INVERT, 0)), p.edge_width, p.dilate,
                p.color, p.color_follow, obj.name, out_path)
            if got is None:
                failed += 1
                continue
            if can_register:
                try:
                    bpy.ops.surfdecal.library_add('EXEC_DEFAULT',
                                                  filepath=got,
                                                  copy_to_library=True)
                except Exception as exc:
                    # 一次失败就别再试了, 但原因必须带出来 ——
                    # 最常见的是 yo的贴花 还没设贴花库目录
                    can_register = False
                    reason.append(str(exc).strip().splitlines()[-1][:120])
            done.append((obj.name, note))
        cache.clear()
        if not done:
            self.report({'ERROR'}, "一张都没烘成功")
            return {'CANCELLED'}
        tail = ", %d 张失败" % failed if failed else ""
        if can_register:
            self.report({'INFO'}, "烘出 %d 张, 已收进贴花库%s"
                        % (len(done), tail))
        else:
            self.report({'WARNING'}, "烘出 %d 张到 %s, 未登记%s%s"
                        % (len(done), outdir, tail,
                           " — " + reason[0] if reason else ""))
        return {'FINISHED'}


class ATLASCUT_OT_adopt(bpy.types.Operator):
    """把所选平面交给 yo的贴花 接管"""
    bl_idname = "atlascut.adopt"
    bl_label = "交给 yo的贴花"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        op = getattr(bpy.ops, "surfdecal", None)
        if op is None or not hasattr(op, "adopt_external"):
            self.report({'ERROR'},
                        "未找到 yo的贴花 的[接管外部平面](需 v5.1.2+)")
            return {'CANCELLED'}
        p = context.scene.atlascut
        try:
            return op.adopt_external('INVOKE_DEFAULT', after=p.adopt_after)
        except Exception as e:
            self.report({'ERROR'}, "接管失败: %s" % e)
            return {'CANCELLED'}


class ATLASCUT_OT_clear_cache(bpy.types.Operator):
    """清除像素缓存(改了图集内容后用)"""
    bl_idname = "atlascut.clear_cache"
    bl_label = "清除像素缓存"

    def execute(self, context):
        _PX_CACHE.clear()
        self.report({'INFO'}, "已清除")
        return {'FINISHED'}


# ============================================================
#  属性
# ============================================================

def sync_targets(p):
    """当前图集出来的、仍由本插件管理的材质。

    只动当前图集 —— 换一张图集不该改上一张的贴花。
    已被 yo的贴花 的 [RGB分离] 接管的材质跳过, 两套接线不互相拆台。
    """
    img = p.atlas
    if img is None:
        return []
    out = []
    for mat in bpy.data.materials:
        if mat.get(AC_MARK) != img.name or not mat.use_nodes:
            continue
        if mat.get("sd_rgb_split"):
            continue
        if mat.node_tree.nodes.get(NODE_DEC) is None:
            continue
        out.append(mat)
    return out


def _tag(mat):
    """节点默认值改完必须打标记, 否则视图不一定重算。"""
    for target in (mat.node_tree, mat):
        try:
            target.update_tag()
        except Exception:
            pass


def _sync_decode(self, context):
    """阈值/颜色: 直接改解码组的输入默认值。"""
    for mat in sync_targets(self):
        n = mat.node_tree.nodes.get(NODE_DEC)
        n.inputs["过渡宽度"].default_value = self.edge_width
        n.inputs["轮廓膨胀"].default_value = self.dilate
        n.inputs["贴花颜色"].default_value = (self.color[0], self.color[1],
                                              self.color[2], 1.0)
        if "色边跟随" in n.inputs:
            n.inputs["色边跟随"].default_value = self.color_follow
        tex = mat.node_tree.nodes.get(NODE_IMG)
        if tex is not None:
            tex.interpolation = self.interp
        _tag(mat)


def _sync_source(self, context):
    """通道/极性: 重接遮罩来源, 并同步材质签名(否则共享判据会错位)。"""
    img = self.atlas
    if img is None:
        return
    if self.invert_mode == 'AUTO':
        _soft, bright = analyze_rect(img, self.channel, 0, 0,
                                     img.size[0], img.size[1])
        invert = not bright
    else:
        invert = (self.invert_mode == 'ON')
    for mat in sync_targets(self):
        wire_channel(mat.node_tree, self.channel)
        n = mat.node_tree.nodes.get(NODE_DEC)
        n.inputs["反相"].default_value = 1.0 if invert else 0.0
        mat[AC_MAT_SIG] = material_signature(img, self.channel, invert,
                                             self.blend)
        _tag(mat)


class AC_Props(bpy.types.PropertyGroup):
    atlas: bpy.props.PointerProperty(
        name="图集", type=bpy.types.Image,
        description="要取图的贴花图集")
    channel: bpy.props.EnumProperty(
        name="遮罩通道", items=CHANNEL_ITEMS, default='R',
        update=_sync_source,
        description="用哪个通道当遮罩; 同一张图的 G/B 里常打包了别的图形")
    invert_mode: bpy.props.EnumProperty(
        name="极性",
        items=[('AUTO', "自动", "按矩形四边的背景亮度判断"),
               ('ON', "反相", "黑底白图形 / 图形在alpha里为1: m = 1 - 通道值"),
               ('OFF', "不反相", "亮底暗图形(如红底绿图): m = 通道值")],
        default='AUTO', update=_sync_source)
    edge_width: bpy.props.FloatProperty(
        name="过渡宽度", default=ALPHA_WIDTH, min=0.002, max=0.5,
        update=_sync_decode,
        description="两个阈值的间距: 越小边缘越硬, 越大越羽化")
    dilate: bpy.props.FloatProperty(
        name="轮廓膨胀", default=0.0, min=-0.3, max=0.3,
        update=_sync_decode,
        description="阈值整体位移: 正=图形变粗, 负=变细")
    color_follow: bpy.props.FloatProperty(
        name="色边跟随", default=1.0, min=0.0, max=1.0, subtype='FACTOR',
        update=_sync_decode,
        description="基础色是否跟着遮罩变暗。软遮罩要开(重建轮廓), "
                    "硬边图集要关(否则边缘像素被压暗, 出现黑边)")
    auto_edge: bpy.props.BoolProperty(
        name="自动匹配边缘", default=True,
        description="取图时按检测到的软/硬边自动设置过渡宽度与色边跟随")
    interp: bpy.props.EnumProperty(
        name="插值",
        items=[('Cubic', "立方", "放大时最平滑, 硬边图集推荐"),
               ('Linear', "线性", "默认双线性, 放大会看到折线"),
               ('Closest', "最近邻", "像素风, 完全不插值"),
               ('Smart', "智能", "放大用立方, 缩小用线性")],
        default='Cubic', update=_sync_decode)
    min_size: bpy.props.FloatProperty(
        name="最小尺寸", default=1.0, min=0.0, unit='LENGTH',
        description="贴花最长边的下限, 避免生成一堆过小的贴花")
    color: bpy.props.FloatVectorProperty(
        name="贴花颜色", subtype='COLOR', size=3, min=0.0, max=1.0,
        default=(1.0, 1.0, 1.0), update=_sync_decode)
    margin_mode: bpy.props.EnumProperty(
        name="边距",
        items=[('AUTO', "自动", "软遮罩外扩留羽化, 硬边内缩半像素防溢色"),
               ('CUSTOM', "手动", "始终使用下面的像素值")],
        default='AUTO')
    margin_px: bpy.props.IntProperty(
        name="边距(px)", default=3, min=-16, max=64,
        description="正=外扩, 负=内缩")
    size_mode: bpy.props.EnumProperty(
        name="尺寸基准",
        items=[('TARGET', "跟目标包围盒", "按所选物体最长边的百分比"),
               ('DENSITY', "固定像素密度", "按 px/m 换算")],
        default='TARGET')
    target_pct: bpy.props.FloatProperty(
        name="占目标比例", default=0.25, min=0.001, max=4.0, subtype='FACTOR',
        description="贴花最长边 = 目标包围盒最长边 × 此值")
    px_per_m: bpy.props.FloatProperty(
        name="像素密度", default=512.0, min=1.0,
        description="每米多少像素; 没有目标时的回退基准")
    grid: bpy.props.IntProperty(
        name="细分格数", default=8, min=1, max=64,
        description="平面的初始网格密度; 吸附时 yo的贴花 还会再细分")
    blend: bpy.props.EnumProperty(
        name="透明混合",
        items=[('HASHED', "抖动", "噪点式半透明, 无排序问题(推荐)"),
               ('CLIP', "剪切", "硬裁切"),
               ('BLEND', "混合", "多层叠加可能排序出错")],
        default='HASHED')
    share_material: bpy.props.BoolProperty(
        name="共享图集材质", default=True,
        description="同一图集+通道+极性的贴花共用一份材质; "
                    "被 yo的贴花 定制过的材质不会被复用")
    after: bpy.props.EnumProperty(
        name="生成后",
        items=[('NONE', "仅生成", "只建平面, 之后手动接管"),
               ('ADOPT', "自动接管", "立即交给 yo的贴花")],
        default='NONE')
    adopt_after: bpy.props.EnumProperty(
        name="接管方式",
        items=[('CONFORM', "吸附到所选", "吸附到选中的目标网格"),
               ('NONE', "仅登记", "只登记为贴花, 不吸附")],
        default='CONFORM')
    px_x: bpy.props.IntProperty(name="X", default=0, min=0)
    px_y: bpy.props.IntProperty(name="Y", default=0, min=0)
    px_w: bpy.props.IntProperty(name="宽", default=256, min=2)
    px_h: bpy.props.IntProperty(name="高", default=256, min=2)


# ============================================================
#  面板
# ============================================================

def draw_body(layout, context, in_image_editor):
    p = context.scene.atlascut

    box = layout.box()
    row = box.row(align=True)
    row.prop(p, "atlas", text="")
    row.operator("atlascut.load", text="", icon='FILEBROWSER')
    if in_image_editor:
        box.operator("atlascut.use_editor_image", icon='IMAGE_REFERENCE')
    if p.atlas is not None and p.atlas.size[0]:
        box.label(text="%d × %d" % (p.atlas.size[0], p.atlas.size[1]),
                  icon='INFO')

    box = layout.box()
    box.label(text="遮罩来源", icon='NODE_TEXTURE')
    box.prop(p, "channel")
    box.prop(p, "invert_mode")
    if p.atlas is not None and p.atlas.size[0]:
        soft, bright = analyze_rect(p.atlas, p.channel, 0, 0,
                                    p.atlas.size[0], p.atlas.size[1])
        box.label(text="整图: %s · 背景%s → 自动%s"
                  % ("软遮罩" if soft else "硬边",
                     "亮" if bright else "暗",
                     "不反相" if bright else "反相"), icon='INFO')

    box = layout.box()
    n = len(sync_targets(p))
    box.label(text="轮廓重建 (实时同步 %d 张)" % n if n else "轮廓重建",
              icon='IPO_EASE_IN_OUT')
    box.prop(p, "auto_edge")
    row = box.row(align=True)
    row.active = not p.auto_edge
    row.prop(p, "edge_width", slider=True)
    box.prop(p, "dilate", slider=True)
    box.prop(p, "color_follow", slider=True)
    box.prop(p, "color", text="")

    box = layout.box()
    box.label(text="取图", icon='UV_FACESEL')
    if in_image_editor:
        box.operator("atlascut.pick_rect", icon='SELECT_SET')
    else:
        box.label(text="拖框请在图像编辑器里用", icon='INFO')
    row = box.row(align=True)
    row.prop(p, "px_x"); row.prop(p, "px_y")
    row = box.row(align=True)
    row.prop(p, "px_w"); row.prop(p, "px_h")
    box.operator("atlascut.cut_manual", icon='ADD')
    box.operator("atlascut.cut_from_uv", icon='UV')
    box.operator("atlascut.cut_grid", icon='MESH_GRID')
    box.operator("atlascut.reframe", icon='FILE_REFRESH')

    box = layout.box()
    box.label(text="生成参数", icon='PREFERENCES')
    box.prop(p, "size_mode", text="")
    if p.size_mode == 'TARGET':
        box.prop(p, "target_pct", slider=True)
        t = pick_target(context)
        box.label(text="参照: %s" % (t.name if t else "无(回退像素密度)"),
                  icon='OBJECT_DATA')
    box.prop(p, "px_per_m")
    box.prop(p, "min_size")
    box.prop(p, "interp", text="插值")
    box.prop(p, "grid")
    row = box.row(align=True)
    row.prop(p, "margin_mode", text="")
    row.prop(p, "margin_px", text="")
    box.prop(p, "blend", text="")
    box.prop(p, "share_material")

    box = layout.box()
    box.label(text="交接", icon='LINKED')
    box.prop(p, "after", text="")
    box.prop(p, "adopt_after", text="")
    box.operator("atlascut.adopt", icon='CHECKMARK')
    box.operator("atlascut.bake_library", icon='RENDERLAYERS')
    box.operator("atlascut.clear_cache", icon='TRASH')


class ATLASCUT_PT_image(bpy.types.Panel):
    bl_label = "图集取图"
    bl_idname = "ATLASCUT_PT_image"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "图集取图"

    def draw(self, context):
        draw_body(self.layout, context, True)


class ATLASCUT_PT_view3d(bpy.types.Panel):
    bl_label = "图集取图"
    bl_idname = "ATLASCUT_PT_view3d"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "图集取图"

    def draw(self, context):
        draw_body(self.layout, context, False)


# ============================================================
#  注册
# ============================================================

classes = (
    AC_Props,
    ATLASCUT_OT_load,
    ATLASCUT_OT_use_editor_image,
    ATLASCUT_OT_pick_rect,
    ATLASCUT_OT_cut_manual,
    ATLASCUT_OT_cut_from_uv,
    ATLASCUT_OT_cut_grid,
    ATLASCUT_OT_reframe,
    ATLASCUT_OT_bake_library,
    ATLASCUT_OT_adopt,
    ATLASCUT_OT_clear_cache,
    ATLASCUT_PT_image,
    ATLASCUT_PT_view3d,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.atlascut = bpy.props.PointerProperty(type=AC_Props)


def unregister():
    global _draw_handle
    if _draw_handle is not None:
        try:
            bpy.types.SpaceImageEditor.draw_handler_remove(
                _draw_handle, 'WINDOW')
        except Exception:
            pass
        _draw_handle = None
    _PX_CACHE.clear()
    if hasattr(bpy.types.Scene, "atlascut"):
        del bpy.types.Scene.atlascut
    for c in reversed(classes):
        try:
            bpy.utils.unregister_class(c)
        except RuntimeError:
            pass


if __name__ == "__main__":
    register()
