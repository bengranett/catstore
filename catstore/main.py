import sys
import os
import logging
import argparse
from catstore import __version__
from .catalogue_store import CatalogueStore


def show(args):
	""" list contents of file """
	for path in args.path:
		with CatalogueStore(path) as cat:
			cat.show(datasets=args.datasets, attrs=args.attrs)

def show_head(args):
	""" print the human readable header """
	for path in args.path:
		with CatalogueStore(path) as cat:
			userblock = cat._h5file.storage.userblock_size
		print(file(path).read(userblock))

def showid(args):
	""" print the file ID """
	for path in args.path:
		with CatalogueStore(path) as cat:
			try:
				digest = cat._hash
			except KeyError as AttributeError:
				digest = 'none'
			print(path, digest)

def verify(args):
	""" run the data validation """
	for path in args.path:
		with CatalogueStore(path) as cat:
			cat.validate_data()

def main():
	parser = argparse.ArgumentParser(prog='CatStore', description="CatStore")

	parser.add_argument('--version', action='version', version=__version__, help='show version and exit')
	parser.add_argument("-v", "--verbose", metavar='n', default=3, type=int, help='verbosity (0,1,2,3)')

	subparser = parser.add_subparsers(title='commands')

	parser_show = subparser.add_parser('show', help='list contents of a catstore file')
	parser_show.add_argument("path", metavar='path', nargs='*', type=str, help="path to catstore file")
	parser_show.add_argument("--datasets", "-d", action='store_true', help="list datasets")
	parser_show.add_argument("--attrs", "-a", action='store_true', help="list attributes")
	parser_show.set_defaults(func=show)

	parser_add = subparser.add_parser('head', help='show the human-readable header of a catstore file')
	parser_add.add_argument("path", metavar='path', nargs='*', type=str, help="path to catstore file")
	parser_add.set_defaults(func=show_head)

	parser_add = subparser.add_parser('id', help='show catstore file ID (hash)')
	parser_add.add_argument("path", metavar='path', nargs='*', type=str, help="path to catstore file")
	parser_add.set_defaults(func=showid)

	parser_verify = subparser.add_parser('verify', help='verify datasets')
	parser_verify.add_argument("path", metavar='path', nargs='*', type=str, help="path to catstore file")
	parser_verify.set_defaults(func=verify)

	args = parser.parse_args()

	if args.verbose > 2:
		logging.basicConfig(level=logging.DEBUG)
	elif args.verbose > 1:
		logging.basicConfig(level=logging.INFO)
	else:
		logging.basicConfig(level=logging.WARNING)

	args.func(args)
