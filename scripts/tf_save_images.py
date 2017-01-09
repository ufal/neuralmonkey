#!/usr/bin/env python3
"""Extract a given image summary from an event file.

Example usage:

    %(prog)s myevents.out example/image --prefix example_

This will extract the contents of the "example/image" summary from the event
file "myevents.out" and write them into the files

example_000000023940.png
example_000000047900.png
...

where the numbers correspond to the steps associated with the images.

The filenames can also be based on the order in which the images appear in the
event file, e.g. "--suffix '{i:06d}.png'" will produce the files

example_000000.png
example_000001.png
...

"""

import argparse
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('event_file', metavar='event-file', help='the event file')
    parser.add_argument('tag', help='the image summary tag')
    parser.add_argument('--prefix', default='image_',
                        help='the image filename prefix (default: %(default)s)')
    parser.add_argument('--suffix', default='{step:012d}.png',
                        help='the image filename suffix formatting string '
                        '(default: %(default)s)')
    args = parser.parse_args()

    i = 0
    for e in tf.train.summary_iterator(args.event_file):
        if e.HasField('summary'):
            for v in e.summary.value:
                if v.HasField('image') and v.tag == args.tag:
                    fname = ('{prefix}' + args.suffix).format(
                        prefix=args.prefix, i=i, step=e.step)

                    with open(fname, 'wb') as f:
                        f.write(v.image.encoded_image_string)

                    i += 1


if __name__ == '__main__':
    main()
