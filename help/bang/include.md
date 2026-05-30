## !include

Import definitions from a `.math` file into the current environment. The `.math` extension may be omitted. `~` is expanded to `$HOME`.

**Usage:** `!include <file>`

**Examples:**
```
> !include ~/mylib.math
included 5 definition(s) from ~/mylib.math
> !include utils
included 3 definition(s) from utils.math
```
