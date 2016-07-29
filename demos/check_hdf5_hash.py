import sys
import pypelid.utils.filetools as ft



for filename in sys.argv[1:]:

    check, hashes = ft.check_hdf5_hash(filename)
    if check:
        print "%s: %s checksum passed :D"%(filename, hashes[0])
    else:
        print "%s: checksum failed :( (read:%s computed:%s)"%(filename, hashes[0], hashes[1])
