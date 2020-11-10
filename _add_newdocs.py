"""
Add docstrings to c-extension modules:
Will create a header file for each module in the current directory, and
fill it with the docstrings.
"""
from collections import defaultdict
import io
import os
srcdir = os.path.join(os.path.dirname(__file__), 'src', 'pnumpy')

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


add_newdoc('pnumpy', 'initialize',
"""
Initialize the module. Replaces all the ufunc inner loops with a new version
using ``PyUFunc_ReplaceLoopBySignature``. If none of the other options are
enabled, the original inner loop function will be called. Will also call
``numpy.setbufsize(8192 * 1024)`` to work around numpy issue 17649.
""")


add_newdoc('pnumpy', 'atop_enable',
"""
enable the atop inner loop implementations.
""")


add_newdoc('pnumpy', 'atop_disable',
"""
disable the atop inner loop implementations.
""")


add_newdoc('pnumpy', "atop_isenabled",
"returns True if atop enabled, else False")


add_newdoc('pnumpy', "thread_enable",
"""
Enable worker threads for inner loops when they are large enough to justify
the extra overhead.
""")


add_newdoc('pnumpy', "thread_disable",
"Disable worker threads")


add_newdoc('pnumpy', "thread_isenabled",
"Returns True if worker threads enabled else False")


add_newdoc('pnumpy', "thread_getworkers",
"Get the number of worker threads")


add_newdoc('pnumpy', "thread_setworkers",
"Set the number of worker threads, return previous value. Must be at least 1.")


add_newdoc('pnumpy', "timer_gettsc",
"Get the time stamp counter")


add_newdoc('pnumpy', "timer_getutc",
"Get the time in utc nanos since unix epoch")


add_newdoc('pnumpy', "cpustring",
"Cpu brand string plus features")


add_newdoc('pnumpy', "oldinit",
"old, deprecated")        


add_newdoc('pnumpy', "ledger_enable",
"""
Enable ledger debuggging. This collects statistics on each run of a loop:
input signature and dimensions, time to execute the loop and more
""")


add_newdoc('pnumpy', "ledger_disable",
"Disable ledger")


add_newdoc('pnumpy', "ledger_isenabled",
"Returns True if ledger enabled else False")


add_newdoc('pnumpy', "ledger_info",
"Return ledger information")


add_newdoc('pnumpy', "recycler_enable",
"Enable recycler to compact memory usage")


add_newdoc('pnumpy', "recycler_disable",
"Disable recycler")


add_newdoc('pnumpy', "recycler_isenabled",
"Returns True if recycler enabled else False")


add_newdoc('pnumpy', "recycler_info",
"Return recycler information")

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
