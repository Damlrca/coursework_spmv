# Copyright (C) 2025 Sadikov Damir
# github.com/Damlrca/coursework_spmv

import sys
import matplotlib.pyplot as plt

class Table:
    def __init__(self, metadata, header, table_lines):
        self.size = len(table_lines)
        self.metadata = metadata.strip()
        self.header = header.strip()
        self.NAME = []
        self.FUNCTION = []
        self.TIME_MS = []
        self.NZ = []
        self.DIFF = []
        self.CONV_TIME = []
        for line in table_lines:
            t = line.split()
            self.NAME.append(t[0])
            self.FUNCTION.append(t[1])
            self.TIME_MS.append(float(t[2]))
            self.NZ.append(int(t[3]))
            self.DIFF.append(t[4])
            self.CONV_TIME.append(t[5])
    def Print(self):
        print("table!")
        print("size =", self.size)
        print("metadata =", self.metadata)
        print("header =", self.header)
        print("NAME =", self.NAME)
        print("FUNCTION =", self.FUNCTION)
        print("TIME_MS =", self.TIME_MS)
        print("NZ =", self.NZ)
        print("DIFF =", self.DIFF)
        print("CONV_TIME =", self.CONV_TIME)
        pass


def parse_tables_from_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    tables = []

    in_table = False
    metadata_passed = False
    header_passed = False
    metadata = ""
    header = ""
    current_table_lines = []

    for line in lines:
        if line.startswith('table_start'):
            in_table = True
            metadata_passed = False
            header_passed = False
            header_passed = False
            metadata = ""
            header = ""
            current_table_lines = []
            continue
        if line.startswith('table_end'):
            in_table = False
            tables.append(Table(metadata, header, current_table_lines))
            continue
        if in_table == False:
            continue
        if metadata_passed == False:
            metadata = line
            metadata_passed = True
            continue
        if header_passed == False:
            header = line
            header_passed = True
            continue
        current_table_lines.append(line)
    return tables

def graph_table(table):
    metadata_split = table.metadata.split()
    mtx_type = metadata_split[3]
    mtx_name = metadata_split[1].split('/')[-1].split('.')[0]
    mtx_threads = metadata_split[5]
    print("drawing graphs for", mtx_name, mtx_type, mtx_threads)

    NZ = table.NZ
    base_nz = NZ[0]
    for i in range(table.size):
        NZ[i] /= base_nz
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 2.7))
    ax1.set_title("mtx = " + mtx_name)
    ax1.set_ylabel('Кол-во ненулевых элем-ов\n(в сравнении с CSR)')
    ax1.bar(table.NAME, NZ)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    mem_plot_name = "./" + mtx_type + "_" + mtx_threads + "/graph;mem;" + mtx_name + ".png"
    plt.savefig(mem_plot_name)
    plt.close()

    fig, ax2 = plt.subplots(1, 1, figsize=(8, 2.7))
    ax2.set_title(table.metadata)
    ax2.set_ylabel('Время (мс)')
    ax2.bar(table.NAME, table.TIME_MS)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.tick_params(axis='x', rotation=90)
    naive_time = table.TIME_MS[0]
    min_time = min(table.TIME_MS)
    min_index = table.TIME_MS.index(min_time)
    min_label = table.NAME[min_index]
    ax2.axhline(naive_time, color='red', linestyle='--', linewidth=1, label=f'naive: {naive_time:.2f} мс')
    ax2.axhline(min_time, color='blue', linestyle='--', linewidth=1, label=f'{min_label}: {min_time:.2f} мс')
    ax2.legend()
    plt.tight_layout()
    time_plot_name = "./" + mtx_type + "_" + mtx_threads + "/graph;" + mtx_type + ";" + mtx_name + ".png"
    plt.savefig(time_plot_name)
    plt.close()

if len(sys.argv) > 1:
    for file_path in sys.argv[1:]:
        tables = parse_tables_from_file(file_path)
        for table in tables:
            #table.Print()
            graph_table(table)
else:
    print("no file_paths were passed as arguments!")
 
