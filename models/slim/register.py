"""Register idea "slim"'s LSCDetect head into ultralytics without forking the repo — same
source-patching trick as models/lwso/register.py (see that file for the full why).

Builds on register_lwso(): the cfg/exp*.yaml architectures reuse SPDConvGroup/C3k2Ghost/
EMA/ECA/DySample/BiFPNCat unchanged from models/lwso/modules.py, so the only NEW classes
parse_model must learn about are LSCDetect, LSCDetectBN and C3k2Rep.

Unlike the lwso/fap/star registrations (which extend base_modules/repeat_modules, i.e.
the *block* channel-inference sets), a Detect-family head is special-cased in two other
spots inside parse_model:

  1. the head frozenset — `elif m in frozenset({Detect, WorldDetect, ...})` — which makes
     parse_model append the per-scale input channel list to args, and
  2. the legacy-flag set — `if m in {Detect, YOLOEDetect, ...}` — which stamps
     `m.legacy = legacy` on the head class (harmless for LSCDetect, which builds its own
     shared cv2/cv3 regardless, but kept consistent with stock Detect handling).

Tested against ultralytics 8.3.x (pinned in requirements.txt); any pattern mismatch
raises RuntimeError immediately (fail fast, no silent breakage).
"""

import re

import ultralytics.nn.tasks as tasks

from models._patch_utils import get_current_parse_model_source, save_parse_model_source
from models.lwso.register import register_lwso

from .modules import C3k2Rep, LSCDetect, LSCDetectBN

_registered = False


def register_slim() -> None:
    """Idempotent. Call once before building any model from a slim YAML."""
    global _registered
    if _registered:
        return

    register_lwso()  # SPDConvGroup/C3k2Ghost/EMA/ECA/DySample/BiFPNCat, reused unchanged

    for cls in (LSCDetect, LSCDetectBN, C3k2Rep):
        setattr(tasks, cls.__name__, cls)

    src = get_current_parse_model_source(tasks)
    substitutions = [
        # 1. head frozenset: parse_model appends [ch[x] for x in f] for these classes
        (
            r"\{Detect, WorldDetect,",
            "{LSCDetect, LSCDetectBN, Detect, WorldDetect,",
        ),
        # 2. legacy-flag set: m.legacy = legacy stamped on Detect-family classes
        (
            r"if m in \{Detect, YOLOEDetect,",
            "if m in {LSCDetect, LSCDetectBN, Detect, YOLOEDetect,",
        ),
        # 3. C3k2Rep (ablation B2) infers channels like C3k2Ghost — add to the
        #    channel-inference set and the repeat set. Anchor on the `frozenset(...{`
        #    opening brace (same anchor register_lwso/register_fap/register_star use) and
        #    prepend right after it, so this COMPOSES regardless of which register_*() ran
        #    first: in the DDP subprocess the order is lwso->fap->star->slim (see
        #    _patch_utils.patch_ddp_registration), and star inserts its own modules just
        #    after that same brace, so any anchor that assumes a specific neighbour (e.g.
        #    "{ immediately followed by C3k2Ghost") breaks. Prepending after the brace does
        #    not. We deliberately do NOT touch the `if m is C3k2 or m is C3k2Ghost:` line:
        #    it only sets c3k=True on m/l/x scales (slim is n-scale), C3k2Rep ignores c3k,
        #    and editing it would clash with register_star's exact-match on that same line.
        (
            r"base_modules = frozenset\(\s*\{",
            "base_modules = frozenset(\n        {\n            C3k2Rep,",
        ),
        (
            r"repeat_modules = frozenset\([^\{]*\{",
            "repeat_modules = frozenset(  # modules with 'repeat' arguments\n        {\n"
            "            C3k2Rep,",
        ),
    ]
    for pattern, replacement in substitutions:
        src, count = re.subn(pattern, replacement, src, count=1)
        if count != 1:
            raise RuntimeError(
                f"register_slim: could not patch parse_model (pattern not found: {pattern!r}). "
                "Your ultralytics version is incompatible; install the range pinned in "
                "requirements.txt (ultralytics>=8.3,<8.4)."
            )

    code = compile(src, tasks.__file__, "exec")
    exec(code, tasks.__dict__)  # rebinds tasks.parse_model to the patched version
    save_parse_model_source(tasks, src)
    _registered = True
