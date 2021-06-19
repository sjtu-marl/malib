import pickle5 as pkl
import io
import collections

_CHUNK_SIZE_ = int(1024 * 1024 * 4)
_PICKLE_PROTOCOL_VER_ = 5


def serialize(obj):
    # FIXME(ming): should handle collection types
    f = io.BytesIO()
    p = pkl.Pickler(f, protocol=_PICKLE_PROTOCOL_VER_)
    p.dump(obj)

    return f.getvalue()


def deserialize(serial_obj):
    return pkl.loads(serial_obj)


def binary_chunks(serial_obj):
    while len(serial_obj):
        # print("length:", len(serial_obj))
        idx = min(len(serial_obj), _CHUNK_SIZE_)
        # print("idx:", idx)
        yield serial_obj[:idx]
        serial_obj = serial_obj[idx:]


def recv_chunks(msgs, name: str, request_fields=None):
    f = io.BytesIO()
    fields = {}
    for msg in msgs:
        f.write(getattr(msg, name))
        if request_fields is None:
            for field in msg.DESCRIPTOR.fields:
                fields[field.name] = getattr(msg, field.name)
        else:
            for field_name in request_fields:
                fields[field_name] = getattr(msg, field_name)
    return f.getvalue(), fields


def serialize_pyplot(figure=None, **kargs):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        raise RuntimeError("Required matplotlib package")

    if figure is None:
        figure = plt.gcf()
    if figure is None:
        raise RuntimeError("None figure passed")
    f = io.BytesIO()
    figure.savefig(f, **kargs)
    return f.getvalue()


def deserialize_image(image_data):
    from PIL import Image

    return Image.open(io.BytesIO(image_data))
