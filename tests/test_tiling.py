import numpy as np

from visdet.tiling import batched_nms, nms, predict_tiled, tile_coords


def test_tiles_cover_image():
    for W, H, tile, ov in [(1920, 1080, 640, 0.2), (1000, 700, 512, 0.0), (500, 400, 640, 0.3), (2000, 1500, 320, 0.3)]:
        cov = np.zeros((H, W), dtype=bool)
        for x, y in tile_coords(W, H, tile, ov):
            assert 0 <= x and 0 <= y and x < W and y < H
            cov[y:y + tile, x:x + tile] = True
        assert cov.all(), (W, H, tile, ov)


def test_tile_count_grows_with_overlap():
    assert len(tile_coords(1920, 1080, 640, 0.3)) >= len(tile_coords(1920, 1080, 640, 0.0))


def test_nms_class_aware():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7])
    cls = np.array([0, 0, 1])
    keep = batched_nms(boxes, scores, cls, 0.5)
    assert sorted(keep.tolist()) == [0, 2]          # le doublon de meme classe disparait, pas celui d'une autre classe
    assert nms(boxes, scores, 0.5).tolist() == [0]


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = np.array(xyxy, float).reshape(-1, 4), np.array(conf, float), np.array(cls, float)


class _Res:
    def __init__(self, *a):
        self.boxes = _Boxes(*a)


def _objects_model(objs):
    """Detecteur factice : renvoie les objets (x1,y1,x2,y2) du repere de l'image d'origine qui tombent dans la tuile.
    Le test encode la position de la tuile dans le pixel (0,0) de chaque crop."""
    class M:
        def predict(self, images, **kw):
            single = isinstance(images, np.ndarray)
            res = []
            for im in ([images] if single else images):
                ox, oy = int(im[0, 0, 0]) * 10, int(im[0, 0, 1]) * 10   # decalage encode (multiples de 10)
                h, w = im.shape[:2]
                b = []
                for (x1, y1, x2, y2) in objs:
                    # coordonnees locales, tronquees au crop (comme le ferait un detecteur qui voit une partie)
                    lx1, ly1, lx2, ly2 = x1 - ox, y1 - oy, x2 - ox, y2 - oy
                    cx1, cy1, cx2, cy2 = max(lx1, 0), max(ly1, 0), min(lx2, w), min(ly2, h)
                    if cx2 - cx1 > 0 and cy2 - cy1 > 0:
                        b.append([cx1, cy1, cx2, cy2])
                res.append(_Res(b, [0.9] * len(b), [0] * len(b)))
            return res
    return M()


def test_predict_tiled_merges_duplicates_and_offsets():
    W, H, tile = 400, 300, 200
    img = np.zeros((H, W, 3), dtype=np.uint8)
    # on encode la position de chaque tuile dans son pixel (0,0) via une image-gradient
    from visdet import tiling
    coords = tiling.tile_coords(W, H, tile, 0.5)
    # image dont le pixel (x,y) vaut (x//10, y//10) en canaux 0/1 => le crop (x0,y0) a pour pixel (0,0) (x0//10, y0//10)
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    img[..., 0] = xs // 10
    img[..., 1] = ys // 10
    objs = [(150, 120, 170, 140)]                 # objet dans la zone de chevauchement : vu par plusieurs tuiles
    boxes, scores, cls = predict_tiled(_objects_model(objs), img, tile=tile, overlap=0.5, merge_iou=0.5)
    assert len(boxes) == 1
    assert np.allclose(boxes[0], [150, 120, 170, 140])
