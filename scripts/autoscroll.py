#!/usr/bin/env python3
"""Auto-scroll a text file for screen recording. Creates test videos for glassrip."""

import argparse
import curses
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description="Display and auto-scroll a text file. Screen-record this to create glassrip test input."
    )
    parser.add_argument("file", help="Text file to scroll through")
    parser.add_argument(
        "--scroll-lines", type=int, default=15,
        help="Lines to advance per scroll step (default: 15, use less than screen height for overlap)"
    )
    parser.add_argument(
        "--pause", type=float, default=2.0,
        help="Seconds to pause between scrolls (default: 2.0)"
    )
    parser.add_argument(
        "--initial-pause", type=float, default=3.0,
        help="Seconds to wait before first scroll (default: 3.0)"
    )
    parser.add_argument(
        "--final-pause", type=float, default=3.0,
        help="Seconds to hold on last page before exit (default: 3.0)"
    )
    parser.add_argument(
        "--no-line-numbers", action="store_true",
        help="Hide line numbers"
    )
    args = parser.parse_args()

    try:
        with open(args.file) as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        print(f"File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    if not lines:
        print("Empty file", file=sys.stderr)
        sys.exit(1)

    curses.wrapper(lambda stdscr: run(stdscr, lines, args))


def run(stdscr, lines, args):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.clear()

    height, width = stdscr.getmaxyx()
    visible = height - 1
    max_offset = max(0, len(lines) - visible)
    show_nums = not args.no_line_numbers
    offset = 0

    draw(stdscr, lines, offset, visible, width, show_nums)
    if wait_or_quit(stdscr, args.initial_pause):
        return

    while offset < max_offset:
        offset = min(offset + args.scroll_lines, max_offset)
        draw(stdscr, lines, offset, visible, width, show_nums)
        if wait_or_quit(stdscr, args.pause):
            return

    wait_or_quit(stdscr, args.final_pause)


def draw(stdscr, lines, offset, visible, width, show_nums):
    stdscr.erase()
    total = len(lines)
    gutter = len(str(total)) + 2 if show_nums else 0

    for i in range(visible):
        line_idx = offset + i
        if line_idx >= total:
            break
        try:
            if show_nums:
                num_str = f"{line_idx + 1:>{gutter - 2}}  "
                stdscr.addstr(i, 0, num_str, curses.A_DIM)
            content = lines[line_idx][: width - gutter - 1]
            stdscr.addstr(i, gutter, content)
        except curses.error:
            pass

    end_line = min(offset + visible, total)
    pct = min(100, int(end_line / max(1, total) * 100))
    status = f" {offset + 1}-{end_line} of {total} ({pct}%)  [q to quit] "
    status = status[: width - 1]
    try:
        stdscr.addstr(visible, 0, status, curses.A_REVERSE)
    except curses.error:
        pass

    stdscr.refresh()


def wait_or_quit(stdscr, duration):
    """Sleep for duration, but check for 'q' keypress every 50ms. Returns True if quit."""
    elapsed = 0.0
    interval = 0.05
    while elapsed < duration:
        key = stdscr.getch()
        if key == ord("q"):
            return True
        time.sleep(interval)
        elapsed += interval
    return False


if __name__ == "__main__":
    main()
