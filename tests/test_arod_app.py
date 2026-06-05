import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_app_module():
    spec = importlib.util.spec_from_file_location("dystream_app", ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_app_defaults_to_arod_student_config():
    app = load_app_module()

    assert app.APP_MODEL_NAME == "AROD"
    assert app.DEFAULT_AROD_CONFIG.endswith(
        "configs/distill/blockwise_stream_distill_cross_fm_mixed_trainval_teacher_gt.yaml"
    )
    assert callable(app.load_arod_models)
