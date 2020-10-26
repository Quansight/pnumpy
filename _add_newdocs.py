"""
Add docstrings to c-extension modules:
Will create a header file for each module in the current directory, and
fill it with the docstrings.
"""
from collections import defaultdict
import io
import os
srcdir = os.path.join(os.path.dirname(__file__), 'src', 'fast_numpy_loops')

def append_header(fid, function, docstring):
	key = function.upper() + "_DOC"
	docstring = docstring.replace('"', '\\"')
	docstring = docstring.replace('\n', '"\n"')
	fid.write("\n")
	fid.write(f'static char {key}[] = "{docstring}";')

headers = defaultdict(io.StringIO)

def add_newdoc(module, function, docstring):
	fid = headers[module.upper() + '.h']
	append_header(fid, function, docstring)	


add_newdoc('fast_numpy_loops', 'initialize',
"""
Initialize the module
""")

add_newdoc('fast_numpy_loops', 'atop_enable',
"""
enable the atop
""")

add_newdoc('fast_numpy_loops', 'atop_disable',
"""
disable the atop
""")


add_newdoc('fast_numpy_loops', "atop_isenabled",
"returns True if atop enabled, else False")


add_newdoc('fast_numpy_loops', "thread_enable",
"enable worker threads")


add_newdoc('fast_numpy_loops', "thread_disable",
"disable worker threads")


add_newdoc('fast_numpy_loops', "thread_isenabled",
"returns True if worker threads enabled else False")


add_newdoc('fast_numpy_loops', "thread_getworkers",
"get the number of worker threads")


add_newdoc('fast_numpy_loops', "thread_setworkers",
"set the number of worker threads, return previous value. Must be at least 1.")


add_newdoc('fast_numpy_loops', "timer_gettsc",
"get the time stamp counter")


add_newdoc('fast_numpy_loops', "timer_getutc",
"get the time in utc nanos since unix epoch")


add_newdoc('fast_numpy_loops', "cpustring",
"cpu brand string plus features")


add_newdoc('fast_numpy_loops', "oldinit",
"old, deprecated")        


add_newdoc('fast_numpy_loops', "ledger_enable",
"enable ledger debuggging")


add_newdoc('fast_numpy_loops', "ledger_disable",
"disable ledger")


add_newdoc('fast_numpy_loops', "ledger_isenabled",
"returns True if ledger enabled else False")


add_newdoc('fast_numpy_loops', "ledger_info",
"return ledger information")


add_newdoc('fast_numpy_loops', "recycler_enable",
"enable recycler debuggging")


add_newdoc('fast_numpy_loops', "recycler_disable",
"disable recycler")


add_newdoc('fast_numpy_loops', "recycler_isenabled",
"returns True if recycler enabled else False")


add_newdoc('fast_numpy_loops', "recycler_info",
"return recycler information")

# Rewrite any of the headers that changed

def main():
	for k, v in headers.items():
		txt2 = ''
		target = os.path.join(srcdir, k)
		txt1 = v.getvalue()
		if os.path.exists(target):
			with open(target) as fid:
				txt2 = fid.read()
		if txt1 != txt2:
			print('writing', target)
			with open(target, 'w') as fid:
				fid.write(txt1)

if __name__ == "__main__":
	main()
