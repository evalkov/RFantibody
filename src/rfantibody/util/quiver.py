#!/usr/bin/env python3
'''
    This file defines the Quiver file class which is used to store PDB files and their associated scores.
    This class is going to be the simplest and quickest implementation of a database for the PDB files and their scores.
    This Quiver implementation will be just a list of PDB lines in a single file, with a tag for each PDB file.

    Later this can be made more sophisticated by using a proper database, but for now this will be the simplest implementation.
'''

import os
import sys
from typing import Iterator, Optional


class Quiver():
    def __init__(self, filename, mode, backend='txt'):
        '''
            filename:   the name of the Quiver file to operate on
            mode:       the mode to open the file in, either 'r' for read-only, or 'w' for write-only
        '''

        self.mode = mode
        self.fn = filename

        self.backend = backend

        self.buffer = {}
        self._tag_offsets = {}

        # Perform mode-specific operations
        if self.mode == 'w' or self.mode == 'r':
            self.tags = self._read_tags()
        else:
            sys.exit(f'Quiver file must be opened in either read or write mode, not {self.mode}')

    def _read_tags(self) -> list:
        '''
            Read the tags from the Quiver file
        '''
        tags = []

        # Check if the file exists
        if not os.path.exists(self.fn):
            return tags

        with open(self.fn, 'r') as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.startswith('QV_TAG'):
                    tag = ''.join(line.split()[1:])
                    tags.append(tag)
                    self._tag_offsets[tag] = offset

        return tags

    def get_tags(self):
        return self.tags

    def size(self):
        return len(self.tags)

    def add_pdb(self, pdb_lines: list, tag: str, score_str=None) -> None:
        '''
            Add a PDB file to the Quiver file
            The Quiver file must be opened in write mode to allow for writing.

            Inputs:
                pdb_lines:  a list of strings, each string is a line of the PDB file
                tag:        a string, the tag to associate with this PDB file
                score_str:  a string, the score to associate with this PDB file
        '''

        if self.mode == 'r':
            # We could in the future have this fail and return False
            sys.exit(f'Quiver file must be opened in write mode to allow for writing.')

        if tag in self.tags:
            # This can be made more sophisticated later
            sys.exit(f'Tag {tag} already exists in this file.')

        # We could eventually have a buffering system here to avoid writing to disk every time
        with open(self.fn, 'a') as f:
            offset = f.tell()
            f.write(f'QV_TAG {tag}\n')
            if score_str is not None:
                f.write(f'QV_SCORE {tag} {score_str}\n')
            f.write(''.join(pdb_lines))
            f.write('\n')
        
        self.tags.append(tag)
        self._tag_offsets[tag] = offset

    def iter_structs(self, include_scores: bool = False) -> Iterator[tuple]:
        '''
            Iterates through all structures in a Quiver file in a single pass.

            Inputs:
                include_scores:
                    If True, yield (tag, pdblines, score_str). Otherwise, yield (tag, pdblines).
        '''

        if self.mode == 'w':
            sys.exit('Quiver file must be opened in read mode to allow iteration.')

        if not os.path.exists(self.fn):
            return

        with open(self.fn, 'r') as f:
            current_tag: Optional[str] = None
            current_lines: list[str] = []
            current_score: Optional[str] = None

            for line in f:
                if line.startswith('QV_TAG'):
                    if current_tag is not None:
                        if include_scores:
                            yield current_tag, current_lines, current_score
                        else:
                            yield current_tag, current_lines
                    current_tag = line.split()[1]
                    current_lines = []
                    current_score = None
                    continue

                if current_tag is None:
                    continue

                if line.startswith('QV_SCORE'):
                    if include_scores:
                        parts = line.split(maxsplit=2)
                        current_score = parts[2].strip() if len(parts) > 2 else ''
                    continue

                current_lines.append(line)

            if current_tag is not None:
                if include_scores:
                    yield current_tag, current_lines, current_score
                else:
                    yield current_tag, current_lines

    def get_pdblines(self, tag: str) -> list:
        '''
            Get the PDB lines associated with the given tag
            The Quiver file must be opened in read mode to allow for reading.
            This function will iterate through the file until it finds the tag, then it will return the PDB lines associated with that tag.

            Inputs:
                tag:    a string, the tag to get the PDB lines for

            Outputs:
                pdblines:  a list of strings, each string is a line of the PDB file
        '''

        if self.mode == 'w':
            # We could in the future have this fail and return False
            sys.exit(f'Quiver file must be opened in read mode to allow for reading.')
        
        if tag not in self._tag_offsets:
            sys.exit(f'Requested tag: {tag} which does not exist')

        with open(self.fn, 'r') as f:
            f.seek(self._tag_offsets[tag])
            first_line = f.readline()
            if not first_line.startswith('QV_TAG') or first_line.split()[1] != tag:
                sys.exit(f'Requested tag: {tag} which does not exist')

            pdb_lines = []
            for line in f:
                if line.startswith('QV_SCORE'):
                    continue
                if line.startswith('QV_TAG'):
                    break
                pdb_lines.append(line)

            return pdb_lines

        # If we get here, we didn't find the tag
        sys.exit(f'Requested tag: {tag} which does not exist')

    def get_struct_list(self, tag_list: list):
        '''
            Get a list of structures from the Quiver file
            The Quiver file must be opened in read mode to allow for reading.

            This is going to be implemented in a more efficient way than just calling get_pdblines for each tag in the list.

            Inputs:
                tag_list:       a list of strings, each string is a tag to get the PDB lines for

            Outputs:
                qv_string:      a string, the Quiver file contents of the requested structures

                found_tags:     a list of strings, each string is a tag that was found in the Quiver file
        '''

        if self.mode == 'w':
            # We could in the future have this fail and return False
            sys.exit(f'Quiver file must be opened in read mode to allow for reading.')

        wanted_tags = set(tag_list)
        found_tags = []
        struct_list = []

        for tag, pdblines, score_str in self.iter_structs(include_scores=True):
            if tag not in wanted_tags:
                continue
            found_tags.append(tag)
            struct_list.append(f'QV_TAG {tag}\n')
            if score_str is not None:
                struct_list.append(f'QV_SCORE {tag} {score_str}\n')
            struct_list.extend(pdblines)

        qv_string = ''.join(struct_list)

        return qv_string, found_tags

    def split(self, ntags: int, outdir: str, prefix: str):
        '''
            Split the Quiver file into multiple Quiver files
            The Quiver file must be opened in read mode to allow for reading.

            Inputs:
                ntags:      an integer, the number of tags to put in each file
                outdir:     a string, the directory to put the new files in
                prefix:     a string, the prefix to use for the new files

            Outputs:
                None
        '''

        if self.mode == 'w':
            # We could in the future have this fail and return False
            sys.exit(f'Quiver file must be opened in read mode to allow for reading.')

        # Make sure the output directory exists
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        # Iterate through the tags and write them to the new files
        struct_idx = 0
        out_idx = 0

        with open(self.fn, 'r') as f:
            for line in f:
                if line.startswith('QV_TAG'):
                    if struct_idx % ntags == 0:
                        if struct_idx != 0:
                            f_out.close()
                        f_out = open(f'{outdir}/{prefix}_{out_idx}.qv', 'w')
                        out_idx += 1
                    struct_idx += 1
                f_out.write(line)
            f_out.close()
