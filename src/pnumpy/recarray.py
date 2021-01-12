import warnings
import numpy as np

__all__ = [
    'recarray_to_colmajor']

from pnumpy._pnumpy import recarray_to_colmajor as _recarray_to_colmajor


#-----------------------------------------------------------------------------------------
def recarray_to_colmajor(item, parallel=True):
    """
    Converts a numpy record array (void type) to a dictionary of numpy arrays, col major

    Returns
    -------
    A dictionary of numpy arrays corresponding to the original numpy record array.

    Examples
    --------
    >>> x=np.array([(1.0, 2, 3, 4, 5, 'this is a long test'), (3.0, 4, 5, 6, 7, 'short'), (30.0, 40, 50, 60, 70, '')],
                dtype=[('x', '<f4'), ('y', '<i2'), ('z', 'i8'),('zz','i8'),('yy','i4'),('str','<S20')])
    >>> item=np.tile(x,100_000)
    >>> mydict = recarray_to_colmajor(item)
    """
    if item.dtype.char == 'V':
        # warnings.warn(f"Converting numpy record array. Performance may suffer.")
        # flip row-major to column-major
        list_types = [*item.dtype.fields.values()]
        success = True
        for t in list_types:
            val = t[0].char
            # if the record type has an object or another record type, we cannot handle
            if val == 'O' or val =='V':
                success = False
                break;

        d={}
        if successs and parallel:
            offsets=[]
            arrays=np.empty(len(item.dtype.fields), dtype='O')
            arrlen = len(item)
            count =0
            for name, v in item.dtype.fields.items():
                offsets.append(v[1])
                arr= np.empty(arrlen, dtype=v[0])
                arrays[count] = arr
                count += 1
                # build dict of names and new arrays
                d[name] = arr

            # Call parallel routine to convert
            _recarray_to_colmajor(item, np.asarray(offsets, dtype=np.int64), arrays);

        else:
            # single thread way
            for name in item.dtype.names:
                d[name] = item[:][name].copy()
        return d

    warnings.warn(f"The array passed was not a numpy record array.")
    return item

