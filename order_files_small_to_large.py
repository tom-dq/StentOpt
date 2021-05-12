import urllib.request
import urllib.error
import os

import click

def is_good_fn(f_url) -> bool:
    return 'It-000' not in f_url and 'opt-' not in f_url

def get_file_size(f_url) -> int:
    req = urllib.request.Request(f_url.strip(), method='HEAD')
    try:
        f = urllib.request.urlopen(req)
        size = int(f.headers['Content-Length'])

    except urllib.error.HTTPError:
        size = -1

    print(size, f_url.strip())
    return size

@click.command()
@click.argument('fn_in', type=click.Path(exists=True))
def order_files(fn_in):

    with open(fn_in) as f_in:
        f_urls = list(f_in.readlines())

    f_no_blank = [f_url for f_url in f_urls if f_url.strip()]
    f_filtered = [f_url for f_url in f_no_blank if is_good_fn(f_url)]
    f_url_sorted = sorted(f_filtered, key=get_file_size)

    fn_out = fn_in + "-ordered"
    with open(fn_out, 'x') as f_out:
        f_out.writelines(f_url_sorted)

        
if __name__ == "__main__":
    order_files()