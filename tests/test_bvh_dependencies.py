"""Optional BVH dependencies fail at conversion time with actionable errors."""

from __future__ import annotations

import pytest

from holosoma_retargeting.data_utils import bvh_dependencies


@pytest.mark.parametrize("missing", ["holosoma_retargeting.data_utils.lafan1", "numpy"])
def test_missing_reader_is_distinguished_from_a_broken_install(monkeypatch, missing):
    failure = ModuleNotFoundError("missing dependency", name=missing)

    def import_missing(_name):
        raise failure

    monkeypatch.setattr(bvh_dependencies, "import_module", import_missing)
    with pytest.raises(ModuleNotFoundError) as caught:
        bvh_dependencies.load_lafan_modules()
    if missing == "numpy":
        assert caught.value is failure
    else:
        assert "optional LAFAN reader" in str(caught.value)
        assert caught.value.__cause__ is failure
