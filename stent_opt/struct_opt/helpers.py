
import psutil

def _add_pmem(m1, m2):
    if not m1 and m2:
        return m2

    if m1 and not m2:
        return m1

    # Overwrite to keep the types
    m_working = m1
    for k, v in m2._asdict().items():
        m_working = m_working._replace(**{k:getattr(m1, k) + v})

    return m_working


def print_memory_use(flag: str):
    p = psutil.Process()
    m_main = p.memory_info()
    m_working = m_main

    for child in p.children(recursive=True):
        m_child = child.memory_info()
        m_working = _add_pmem(m_working, m_child)

    def print_output(prefix, m):
        bits = ['peak_wset', 'peak_paged_pool', 'peak_nonpaged_pool']
        out_bits = [flag, prefix, p.name()]
        for b in bits:
            val = getattr(m, b) / (1024**3)
            out_bits.append(f"{b}: {val:1.2g} GB")

        print("\t".join(out_bits))

    print_output("Main", m_main)
    print_output("All ", m_working)

if __name__ == "__main__":
    print_memory_use("testing")

