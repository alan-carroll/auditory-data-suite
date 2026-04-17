"""
Compatibility shim. Everything that used to live here has moved to the
modules below; this file re-exports so existing `afunc.foo` /
`from analysis_functions import foo` call sites keep working. Delete
once callers are migrated.
"""
from dialogs import *          # noqa: F401,F403
from brainware import *        # noqa: F401,F403
from tc_analysis import *      # noqa: F401,F403
from geometry import *         # noqa: F401,F403
from final_file import *       # noqa: F401,F403
from analysis_admin import *   # noqa: F401,F403