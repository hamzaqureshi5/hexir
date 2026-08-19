#!/usr/bin/env python3
"""Read a .hxb module and print what is in it.

The format is defined by runtime/include/hexir_runtime/{module,program}.h --
this is a second implementation of that reader, which is the point: if the two
disagree, one of them is wrong, and a Python one is much easier to argue with.

    tools/hxb-dump.py model.hxb              # header, sections, program
    tools/hxb-dump.py model.hxb --rodata     # also the constant values
    tools/hxb-dump.py model.hxb --image k    # write kernel k's device image out
"""

import argparse
import struct
import sys

MAGIC = b"HEXIRMOD"
VERSION = 3

SECTION_KIND = {1: "symbols", 2: "program", 3: "rodata", 4: "executables"}
KERNEL_KIND = {1: "matmul", 2: "add", 3: "relu"}
DEVICE_KIND = {0: "cpu", 1: "cuda"}

# kind -> (name, operand names). None means "the rest are slots".
COMMANDS = {
    0: ("end", []),
    1: ("alloc", ["slot", "bytes"]),
    2: ("const", ["slot", "rodata_offset", "bytes"]),
    3: ("dispatch", ["executable", "argc", None]),
    4: ("print", ["slot", "rows", "cols"]),
}

HEADER = "<8sIIII"                  # magic, version, flags, count, reserved
SECTION = "<IIQQ"                   # kind, flags, offset, size
SYMBOL = "<32sII"                   # name, program_offset, reserved
EXEC_HEADER = "<II"                 # count, reserved
EXEC_ENTRY = "<32sIIIIIIIIIIII"     # name, kind, device, m, n, k, elem_size,
                                    # image_offset, image_size,
                                    # grid_x, grid_y, block_x, block_y


def die(message):
    sys.exit("hxb-dump: " + message)


def image_kind(blob):
    """Device images are self-describing; say which flavour this is."""
    if blob[:4] == b"\x7fELF":
        return "ELF (cubin)"
    if blob[:4] == b"\x50\xed\x55\xba":  # 0xBA55ED50
        return "fatbin"
    if blob[:2] == b"//" or b".target" in blob[:256]:
        return "PTX (text)"
    return "unknown"


def read_sections(data):
    if len(data) < struct.calcsize(HEADER):
        die("file is too short to hold a header")
    magic, version, flags, count, _ = struct.unpack_from(HEADER, data, 0)
    if magic != MAGIC:
        die("bad magic %r, this is not a hexir module" % magic)
    if version != VERSION:
        print("warning: version %d, this tool understands %d"
              % (version, VERSION), file=sys.stderr)

    sections = {}
    order = []
    base = struct.calcsize(HEADER)
    size = struct.calcsize(SECTION)
    for i in range(count):
        kind, sflags, offset, length = struct.unpack_from(SECTION, data, base + i * size)
        # Offsets come off disk; refuse to index outside the file.
        if offset > len(data) or offset + length > len(data):
            die("section %d runs past the end of the file" % i)
        sections[kind] = (offset, length)
        order.append((kind, sflags, offset, length))
    return version, flags, order, sections


def dump_program(data, offset, length, symbols):
    """Walk the command list. Entry points are marked where they start."""
    starts = {program_offset: name for name, program_offset in symbols}
    end = offset + length
    pc = offset
    print("\nprogram")
    while pc < end:
        if pc - offset in starts:
            print("  %s:" % starts[pc - offset])
        kind, argc = struct.unpack_from("<II", data, pc)
        pc += 8
        if pc + argc * 8 > end:
            die("command at %d claims %d operands and runs off the end"
                % (pc - offset, argc))
        operands = list(struct.unpack_from("<%dQ" % argc, data, pc)) if argc else []
        pc += argc * 8

        name, argnames = COMMANDS.get(kind, ("op%d" % kind, []))
        parts = []
        for i, value in enumerate(operands):
            label = argnames[i] if i < len(argnames) and argnames[i] else None
            parts.append("%s=%d" % (label, value) if label else "%%%d" % value)
        print("    %-9s %s" % (name, " ".join(parts)))
        if kind == 0:
            break


def main():
    parser = argparse.ArgumentParser(description="inspect a .hxb module")
    parser.add_argument("path")
    parser.add_argument("--rodata", action="store_true",
                        help="print constant data as f64")
    parser.add_argument("--image", metavar="KERNEL",
                        help="write that kernel's device image to <KERNEL>.bin")
    args = parser.parse_args()

    data = open(args.path, "rb").read()
    version, flags, order, sections = read_sections(data)

    print("%s  (%d bytes)" % (args.path, len(data)))
    print("version %d, flags 0x%x, %d sections" % (version, flags, len(order)))
    for kind, sflags, offset, length in order:
        print("  %-12s offset=%-8d size=%d"
              % (SECTION_KIND.get(kind, "kind%d" % kind), offset, length))

    symbols = []
    if 1 in sections:
        offset, length = sections[1]
        size = struct.calcsize(SYMBOL)
        print("\nsymbols")
        for i in range(length // size):
            name, program_offset, _ = struct.unpack_from(SYMBOL, data, offset + i * size)
            name = name.rstrip(b"\0").decode()
            symbols.append((name, program_offset))
            print("  %-20s program+%d" % (name, program_offset))

    if 2 in sections:
        dump_program(data, sections[2][0], sections[2][1], symbols)

    if 3 in sections:
        offset, length = sections[3]
        print("\nrodata  %d bytes (%d f64)" % (length, length // 8))
        if args.rodata and length:
            values = struct.unpack_from("<%dd" % (length // 8), data, offset)
            for i in range(0, len(values), 4):
                print("    +%-5d %s" % (i * 8, "  ".join("%g" % v for v in values[i:i + 4])))

    if 4 in sections:
        offset, length = sections[4]
        count, _ = struct.unpack_from(EXEC_HEADER, data, offset)
        size = struct.calcsize(EXEC_ENTRY)
        base = offset + struct.calcsize(EXEC_HEADER)
        print("\nexecutables  (%d)" % count)
        for i in range(count):
            (name, kind, device, m, n, k, elem, image_offset, image_size,
             grid_x, grid_y, block_x, block_y) = struct.unpack_from(
                 EXEC_ENTRY, data, base + i * size)
            name = name.rstrip(b"\0").decode()
            print("  %-16s %-6s %-6s  %dx%dx%d  elem=%dB"
                  % (name, KERNEL_KIND.get(kind, "?"),
                     DEVICE_KIND.get(device, "?"), m, n, k, elem))
            if image_size:
                blob = data[offset + image_offset:offset + image_offset + image_size]
                print("    %-14s %d bytes, %s   grid=(%d,%d) block=(%d,%d)"
                      % ("device image", image_size, image_kind(blob),
                         grid_x, grid_y, block_x, block_y))
                if args.image == name:
                    out = name + ".bin"
                    open(out, "wb").write(blob)
                    print("    wrote %s (try: cuobjdump --dump-sass %s)" % (out, out))
            else:
                print("    %-14s none -- the runtime supplies the body"
                      % "device image")


if __name__ == "__main__":
    main()
