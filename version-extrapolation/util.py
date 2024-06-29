from collections import defaultdict
from inspect import signature
from itertools import chain, combinations
import re
from typing import Iterable, Mapping


def powerset(iterable):
    """ Returns the powerset of an iterable. """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def filter_dict(dict_to_filter, thing_with_kwargs):
    """ Filter a dictionary to only include keys that are valid arguments to a function.
        see https://stackoverflow.com/a/44052550/3769237
    """
    sig = signature(thing_with_kwargs)
    filter_keys = [param.name for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
    filtered_dict = { k: v for k, v in dict_to_filter.items() if k in filter_keys }
    return filtered_dict


def create_version_table(data):
    versions = data.groupby(["parent", "parent version", "child"])["child version"].agg(set)
    versions = versions.to_dict()
    known_versions = defaultdict(set)
        
    for (parent, parent_version, child), child_versions in versions.items():
        known_versions[parent].add(parent_version)
        known_versions[child].update(child_versions)
    
    return known_versions


class Version:
    # reimplemenation of distutils.version.LooseVersion since distutils is deprecated
    # also adapted to handle comparing between mismatched versions like '1.5-develop' and '1.4.3'

    component_re = re.compile(r'(\d+ | [a-z]+ | \.)', re.VERBOSE)

    def __init__(self, vstring=None):
        if vstring:
            self.parse(vstring)
    
    def parse(self, vstring):
        self.vstring = vstring
        components = [x for x in self.component_re.split(vstring)
                              if x and x != '.']
        for i, obj in enumerate(components):
            try:
                components[i] = int(obj)
            except ValueError:
                pass

        self.version = components

    def __str__ (self):
        return self.vstring

    def __repr__ (self):
        return "Version ('%s')" % str(self)

    def _cmp (self, other):
        if isinstance(other, str):
            other = Version(other)
        elif not isinstance(other, Version):
            return NotImplemented
        
        for idx in range(min(len(self.version), len(other.version))):
            if type(self.version[idx]) != type(other.version[idx]):
                self.version[idx] = str(self.version[idx])
                other.version[idx] = str(other.version[idx])

        if self.version == other.version:
            return 0
        if self.version < other.version:
            return -1
        if self.version > other.version:
            return 1
        
    def __eq__(self, other):
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c == 0

    def __lt__(self, other):
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c < 0

    def __le__(self, other):
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c <= 0

    def __gt__(self, other):
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c > 0

    def __ge__(self, other):
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c >= 0

def encode_versions(versions: Iterable[str], offset: int = 0) -> Mapping[str, int]:
    """ Ordinally encode versions based on distutils.version.LooseVersion.
        LooseVersion can handle most things, except for things like 'master', 'main', 'develop', 'latest'.
        We want to encode these as the highest version numbers in the order of 'master', 'main', 'develop', 'latest'.
    """
    to_remove = ['master', 'main', 'develop', 'latest']
    removed = [v for v in versions if v in to_remove]
    versions = [v for v in versions if v not in to_remove]
    versions.sort(key=Version)

    encoded = { v: idx for idx, v in enumerate(versions, start=offset) }
    for r in removed:
        encoded[r] = len(encoded) + offset
    
    return encoded