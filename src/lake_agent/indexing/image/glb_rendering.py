from __future__ import annotations

import io
import hashlib
import math
import os
import re
from pathlib import Path
from typing import Any


def render_glb_to_6_views(file_path: str) -> list[bytes]:
    """Render a GLB model into six canonical PNG views entirely in memory.

    Returns PNG-encoded bytes in this order:
    Front, Back, Left, Right, Top, Bottom.
    """

    try:
        import numpy as np
        from PIL import Image
        import trimesh
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "GLB rendering requires numpy, Pillow, and trimesh. "
            "Install with: pip install numpy Pillow trimesh pyrender"
        ) from exc

    # pyrender reads the OpenGL platform at import time. EGL works for many
    # Linux headless runners; callers may override with PYOPENGL_PLATFORM=osmesa.
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    try:
        import pyrender
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "GLB rendering requires pyrender. Install with: pip install pyrender"
        ) from exc

    path = Path(file_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".glb":
        raise ValueError(f"Expected a .glb file, got: {path.suffix or '<no extension>'}")

    width = 1024
    height = 1024
    yfov = math.radians(45.0)
    aspect_ratio = width / height

    try:
        trimesh_scene = _load_as_trimesh_scene(path, trimesh)
        bounds = trimesh_scene.bounds
        if bounds is None:
            raise ValueError("GLB scene does not expose valid bounds.")

        bounds_array = np.asarray(bounds, dtype=float)
        if bounds_array.shape != (2, 3) or not np.isfinite(bounds_array).all():
            raise ValueError("GLB scene has invalid or non-finite bounds.")

        center = bounds_array.mean(axis=0)
        extents = bounds_array[1] - bounds_array[0]
        radius = float(np.linalg.norm(extents) / 2.0)
        if radius <= 0.0:
            radius = 1.0

        distance = _camera_distance_for_bounds(
            radius=radius,
            yfov=yfov,
            aspect_ratio=aspect_ratio,
            padding=1.35,
        )

        scene = pyrender.Scene.from_trimesh_scene(
            trimesh_scene,
            bg_color=[255, 255, 255, 0],
            ambient_light=[0.38, 0.38, 0.38],
        )
        camera = pyrender.PerspectiveCamera(
            yfov=yfov,
            aspectRatio=aspect_ratio,
            znear=max(0.01, distance - radius * 3.0),
            zfar=distance + radius * 3.0,
        )
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

        try:
            images: list[bytes] = []
            for eye_offset, up in _view_directions(distance):
                eye = center + np.asarray(eye_offset, dtype=float)
                pose = _look_at_camera_pose(eye=eye, target=center, up=np.asarray(up, dtype=float), np=np)

                camera_node = scene.add(camera, pose=pose)
                # Directional light shares the camera pose so the visible face is
                # lit from the viewer side, avoiding dark silhouettes.
                light_node = scene.add(light, pose=pose)
                try:
                    color, _depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
                finally:
                    scene.remove_node(camera_node)
                    scene.remove_node(light_node)

                buffer = io.BytesIO()
                Image.fromarray(color).save(buffer, format="PNG")
                images.append(buffer.getvalue())
            return images
        finally:
            renderer.delete()
    except Exception as exc:
        if isinstance(exc, (FileNotFoundError, ValueError, RuntimeError)):
            raise
        raise RuntimeError(f"Failed to render GLB model '{path}': {exc}") from exc


def render_glb_to_6_view_files(
    file_path: str,
    output_root: str | Path,
) -> list[Path]:
    """Render a GLB into one folder containing six canonical PNG view files.

    Output layout:
        output_root/<model_stem>/front.png
        output_root/<model_stem>/back.png
        output_root/<model_stem>/left.png
        output_root/<model_stem>/right.png
        output_root/<model_stem>/top.png
        output_root/<model_stem>/bottom.png
    """

    path = Path(file_path).expanduser().resolve()
    model_dir = Path(output_root).expanduser().resolve() / _safe_folder_name(path.stem)
    model_dir.mkdir(parents=True, exist_ok=True)

    rendered_views = render_glb_to_6_views(os.fspath(path))
    output_paths: list[Path] = []
    for view_name, image_bytes in zip(_VIEW_NAMES, rendered_views, strict=True):
        output_path = model_dir / f"{view_name}.png"
        output_path.write_bytes(image_bytes)
        output_paths.append(output_path)
    return output_paths


def render_glb_to_6_image_results(
    file_path: str,
    *,
    output_root: str | Path | None = None,
    ocr_extractor: Any | None = None,
    vlm_enricher: Any | None = None,
) -> list[Any]:
    """Render a GLB and pass each generated PNG through LakeAgent image processing.

    The returned items are ImageIndexResult objects. OCR and VLM processing are
    optional and use the existing image extractors/enrichers when provided.
    """

    from PIL import Image

    from lake_agent.domain.indexing_models import ImageIndexResult

    path = Path(file_path).expanduser().resolve()
    view_paths: list[Path] | None = None
    if output_root is None:
        view_bytes = render_glb_to_6_views(os.fspath(path))
    else:
        view_paths = render_glb_to_6_view_files(os.fspath(path), output_root)
        view_bytes = [view_path.read_bytes() for view_path in view_paths]
    stem = path.stem

    results: list[ImageIndexResult] = []
    image_payloads: list[tuple[str, bytes]] = []
    for index, (view_name, image_bytes) in enumerate(zip(_VIEW_NAMES, view_bytes, strict=True)):
        view_path = view_paths[index] if view_paths is not None else None
        filename = view_path.name if view_path is not None else f"{stem}_{view_name}.png"
        source_id = _stable_id(f"{path.as_posix()}:{view_name}", prefix="glbview")
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
            color_mode = image.mode
            has_alpha = color_mode in {"RGBA", "LA", "PA"} or "transparency" in image.info

        relative_path = (
            view_path.as_posix()
            if view_path is not None
            else f"{path.name}#{view_name}"
        )
        result = ImageIndexResult(
            source_id=source_id,
            relative_path=relative_path,
            filename=filename,
            file_format="png",
            width=width,
            height=height,
            color_mode=color_mode,
            has_alpha=has_alpha,
            is_animated=False,
            frame_count=1,
            parse_warnings=[],
            file_keywords=["glb", "3d model", view_name],
        )
        result.file_search_text = _build_rendered_view_search_text(result, view_name)
        results.append(result)
        image_payloads.append((filename, image_bytes))

    if ocr_extractor is not None:
        if view_paths is not None:
            sections_by_source = ocr_extractor.extract_sections_batch(
                view_paths,
                source_ids=[result.source_id for result in results],
            )
        else:
            sections_by_source = ocr_extractor.extract_sections_bytes_batch(
                image_payloads,
                source_ids=[result.source_id for result in results],
            )
        for result in results:
            result.sections.extend(sections_by_source.get(result.source_id, []))

    if vlm_enricher is not None:
        if view_paths is not None:
            vlm_enricher.enrich_batch(view_paths, results)
        else:
            vlm_enricher.enrich_bytes_batch(image_payloads, results)

    for result, view_name in zip(results, _VIEW_NAMES, strict=True):
        result.file_search_text = _build_rendered_view_search_text(result, view_name)
    return results


def _load_as_trimesh_scene(path: Path, trimesh: Any) -> Any:
    try:
        loaded = trimesh.load(path, force="scene", process=False)
    except Exception as exc:
        raise ValueError(f"Unable to load GLB file '{path}': {exc}") from exc

    if isinstance(loaded, trimesh.Trimesh):
        loaded = trimesh.Scene(loaded)
    if not isinstance(loaded, trimesh.Scene):
        raise ValueError(f"Unsupported GLB payload type: {type(loaded).__name__}")
    if not loaded.geometry:
        raise ValueError("GLB scene contains no renderable geometry.")
    return loaded


def _camera_distance_for_bounds(
    *,
    radius: float,
    yfov: float,
    aspect_ratio: float,
    padding: float,
) -> float:
    vertical_half_fov = yfov / 2.0
    horizontal_half_fov = math.atan(math.tan(vertical_half_fov) * aspect_ratio)
    limiting_half_fov = min(vertical_half_fov, horizontal_half_fov)
    return (radius / math.sin(limiting_half_fov)) * padding


def _view_directions(distance: float) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    return [
        # Front: camera on +Z, looking toward the object along world -Z.
        ((0.0, 0.0, distance), (0.0, 1.0, 0.0)),
        # Back: equivalent to rotating the front camera 180 degrees around Y.
        ((0.0, 0.0, -distance), (0.0, 1.0, 0.0)),
        # Left/Right: +/-90 degrees around Y relative to the front view.
        ((-distance, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((distance, 0.0, 0.0), (0.0, 1.0, 0.0)),
        # Top/Bottom: rotate around X; use Z as the image-up axis to avoid roll ambiguity.
        ((0.0, distance, 0.0), (0.0, 0.0, -1.0)),
        ((0.0, -distance, 0.0), (0.0, 0.0, 1.0)),
    ]


def _look_at_camera_pose(*, eye: Any, target: Any, up: Any, np: Any) -> Any:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    # pyrender cameras look along local -Z, so camera +Z points backward from
    # target to eye in world coordinates.
    z_axis = -forward
    x_axis = np.cross(up, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        raise ValueError("Camera up vector is parallel to the viewing direction.")
    x_axis = x_axis / x_norm
    y_axis = np.cross(z_axis, x_axis)

    pose = np.eye(4, dtype=float)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = eye
    return pose


def _build_rendered_view_search_text(result: Any, view_name: str) -> str:
    parts = [
        result.filename,
        result.relative_path,
        "glb",
        "3d model",
        f"{view_name} view",
        result.file_format,
        f"{result.width}x{result.height}",
        result.color_mode,
    ]
    if result.file_summary:
        parts.append(result.file_summary)
    if result.file_keywords:
        parts.append(", ".join(result.file_keywords))
    for section in result.sections[:2]:
        if section.heading:
            parts.append(section.heading)
        if section.content:
            parts.append(section.content[:600])
    return "\n".join(part for part in parts if part).strip()


def _stable_id(value: str, *, prefix: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _safe_folder_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "model"


_VIEW_NAMES = ["front", "back", "left", "right", "top", "bottom"]
