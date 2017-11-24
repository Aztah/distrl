import matplotlib
import csv
import statistics
import os
import bisect
import time

def write_csv(run_code, log, comm_rounds=False):
    file_name = run_code + ("cr" if comm_rounds else "") + ".csv"
    if log is not None:
        try:
            with open(file_name, 'a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=';')
                csv_writer.writerow(log)
        except PermissionError:
            print("Permission error. CSV write failed:", log)


def write_csv_final(file_name, final_episode, worker_hosts=None, chief=False, comm_rounds=0):
    new_filename = file_name + "=" + str(final_episode) + "ep.csv"
    os.rename(file_name + ".csv", new_filename)
    # with open(file_name + ".csv", 'r', newline='') as infile, open(new_filename, 'w', newline='') as outfile:
    #     reader = csv.reader(infile, delimiter=';')
    #     writer = csv.writer(outfile, delimiter=';')
    #     i = 0
    #     for row in reader:
    #         writer.writerow(row + ([] if i >= len(round_log) else round_log[i]))
    #         i += 1
    #
    #     for a in round_log[i:]:
    #         writer.writerow([None, None, None, None] + a)
    # os.remove(file_name + ".csv")
    if chief:
        if all([host.find("localhost") >= 0 for host in worker_hosts]):
            data = []
            f1 = file_name.split("(w")[0]
            files = []
            print("All localhost. Chief combining files")
            while len(files) < len(worker_hosts):
                files = list(filter(lambda f: f.find(f1) >= 0 and f.find(")=") >= 0, os.listdir('.')))
            print("Files to combine:", files)
            for file in files:
                buffer = []
                # TODO fix permission error: PermissionError: [Errno 13] Permission denied
                for attempt in range(100):
                    try:
                        with open(file, 'r', newline='') as infile:
                            reader = csv.reader(infile, delimiter=';')
                            buffer = [row for row in reader]
                        bisect.insort_left(data, buffer)
                    except PermissionError as e:
                        print("Could not open file ", file, " Permission error(", attempt, "):", e.strerror, sep='')
                        time.sleep(5)
                        continue
                    else:
                        break
                else:
                    print("All failed. Some files will not be combined.")
            data_len = [len(x) for x in data]
            print("Data of length", data_len, "\n", data)
            summary_name = "{}-avg-{}-med-{}-sdv-{}-min-{}-max-{}-cr-{}.csv"\
                .format(file_name.split("(")[0], round(statistics.mean(data_len)),
                        round(statistics.median(data_len)),
                        (round(statistics.stdev(data_len), 1) if len(data_len) > 1 else 0),
                        min(data_len), max(data_len), int(comm_rounds))
            with open(summary_name, 'w', newline='') as csv_file:
                i = 0
                csv_writer = csv.writer(csv_file, delimiter=';')
                while i < max(data_len):
                    writing = []
                    for j in range(len(data)):
                        if len(data[j]) <= i:
                            # This needs to be changed if data length changes
                            writing += [i, 200, 200, 0, 0, None]
                        else:
                            writing += data[j][i] + [None]
                    # writing = list(itertools.chain.from_iterable([[i, 200, 200, 0, None] if len(run) > i else run[i] + [None] for run in data]))
                    if i == 0:
                        writing += ["avg_reward", "avg_avg_reward"]
                    else:
                        # This needs to be changed if data length changes
                        writing += [statistics.mean([float(x) for x in writing[1::6]]), statistics.mean([float(x) for x in writing[2::6]])]
                    csv_writer.writerow(writing)
                    i += 1
            [os.remove(f) for f in files]
        else:
            print("Some hosts are not localhost, not combining files" + worker_hosts)

    print("Results saved in:  ", new_filename, sep='')

def merge_cr(identifier):
    pass

def plot_experiment(identifier):
    data = []
    pass