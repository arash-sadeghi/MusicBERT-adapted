import miditoolkit
import signal
import math
import time
from multiprocessing import Pool, Lock, Manager
import io
import random



pos_resolution = 16  # per beat (quarter note)
bar_max = 256
velocity_quant = 4
tempo_quant = 12  # 2 ** (1 / 12)
min_tempo = 16
max_tempo = 256
duration_max = 8  # 2 ** 8 * beat
max_ts_denominator = 6  # x/1 x/2 x/4 ... x/64
max_notes_per_bar = 2  # 1/64 ... 128/64
beat_note_factor = 4  # In MIDI format a note is always 4 beats
deduplicate = True
filter_symbolic = False
filter_symbolic_ppl = 16
trunc_pos = 2 ** 16  # approx 30 minutes (1024 measures)
sample_len_max = 1000  # window length max
sample_overlap_rate = 4
ts_filter = False
pool_num = 24
max_inst = 127
max_pitch = 127
max_velocity = 127

data_zip = None
output_file = None

lock_file = Lock()
lock_write = Lock()


ts_dict = dict()
ts_list = list()
for i in range(0, max_ts_denominator + 1):  # 1 ~ 64
    for j in range(1, ((2 ** i) * max_notes_per_bar) + 1):
        ts_dict[(j, 2 ** i)] = len(ts_dict)
        ts_list.append((j, 2 ** i))
dur_enc = list()
dur_dec = list()
for i in range(duration_max):
    for j in range(pos_resolution):
        dur_dec.append(len(dur_enc))
        for k in range(2 ** i):
            dur_enc.append(len(dur_dec) - 1)

def t2e(x):
    assert x in ts_dict, 'unsupported time signature: ' + str(x)
    return ts_dict[x]

def e2t(x):
    return ts_list[x]

def d2e(x):
    return dur_enc[x] if x < len(dur_enc) else dur_enc[-1]

def e2d(x):
    return dur_dec[x] if x < len(dur_dec) else dur_dec[-1]

def v2e(x):
    return x // velocity_quant

def e2v(x):
    return (x * velocity_quant) + (velocity_quant // 2)

def b2e(x):
    x = max(x, min_tempo)
    x = min(x, max_tempo)
    x = x / min_tempo
    e = round(math.log2(x) * tempo_quant)
    return e

def e2b(x):
    return 2 ** (x / tempo_quant) * min_tempo

def time_signature_reduce(numerator, denominator):
    # reduction (when denominator is too large)
    while denominator > 2 ** max_ts_denominator and denominator % 2 == 0 and numerator % 2 == 0:
        denominator //= 2
        numerator //= 2
    # decomposition (when length of a bar exceed max_notes_per_bar)
    while numerator > max_notes_per_bar * denominator:
        for i in range(2, numerator + 1):
            if numerator % i == 0:
                numerator //= i
                break
    return numerator, denominator

def writer(output_file, output_str_list):
    # note: parameter "file_name" is reserved for patching
    with open(output_file, 'a') as f:
        for output_str in output_str_list:
            f.write(output_str + '\n')

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, value, traceback):
        signal.alarm(0)

def MIDI_to_encoding(midi_obj):
    def time_to_pos(t):
        return round(t * pos_resolution / midi_obj.ticks_per_beat)
    notes_start_pos = [time_to_pos(j.start)
                       for i in midi_obj.instruments for j in i.notes]
    if len(notes_start_pos) == 0:
        return list()
    max_pos = min(max(notes_start_pos) + 1, trunc_pos)
    pos_to_info = [[None for _ in range(4)] for _ in range(
        max_pos)]  # (Measure, TimeSig, Pos, Tempo)
    tsc = midi_obj.time_signature_changes
    tpc = midi_obj.tempo_changes
    for i in range(len(tsc)):
        for j in range(time_to_pos(tsc[i].time), time_to_pos(tsc[i + 1].time) if i < len(tsc) - 1 else max_pos):
            if j < len(pos_to_info):
                pos_to_info[j][1] = t2e(time_signature_reduce(
                    tsc[i].numerator, tsc[i].denominator))
    for i in range(len(tpc)):
        for j in range(time_to_pos(tpc[i].time), time_to_pos(tpc[i + 1].time) if i < len(tpc) - 1 else max_pos):
            if j < len(pos_to_info):
                pos_to_info[j][3] = b2e(tpc[i].tempo)
    for j in range(len(pos_to_info)):
        if pos_to_info[j][1] is None:
            # MIDI default time signature
            pos_to_info[j][1] = t2e(time_signature_reduce(4, 4))
        if pos_to_info[j][3] is None:
            pos_to_info[j][3] = b2e(120.0)  # MIDI default tempo (BPM)
    cnt = 0
    bar = 0
    measure_length = None
    for j in range(len(pos_to_info)):
        ts = e2t(pos_to_info[j][1])
        if cnt == 0:
            measure_length = ts[0] * beat_note_factor * pos_resolution // ts[1]
        pos_to_info[j][0] = bar
        pos_to_info[j][2] = cnt
        cnt += 1
        if cnt >= measure_length:
            assert cnt == measure_length, 'invalid time signature change: pos = {}'.format(
                j)
            cnt -= measure_length
            bar += 1
    encoding = []
    start_distribution = [0] * pos_resolution
    for inst in midi_obj.instruments:
        for note in inst.notes:
            if time_to_pos(note.start) >= trunc_pos:
                continue
            start_distribution[time_to_pos(note.start) % pos_resolution] += 1
            info = pos_to_info[time_to_pos(note.start)]
            encoding.append((info[0], info[2], max_inst + 1 if inst.is_drum else inst.program, note.pitch + max_pitch +
                             1 if inst.is_drum else note.pitch, d2e(time_to_pos(note.end) - time_to_pos(note.start)), v2e(note.velocity), info[1], info[3]))
    if len(encoding) == 0:
        return list()
    tot = sum(start_distribution)
    start_ppl = 2 ** sum((0 if x == 0 else -(x / tot) *
                          math.log2((x / tot)) for x in start_distribution))
    # filter unaligned music
    if filter_symbolic:
        assert start_ppl <= filter_symbolic_ppl, 'filtered out by the symbolic filter: ppl = {:.2f}'.format(
            start_ppl)
    encoding.sort()
    return encoding

def F(file_name):
    try_times = 1
    midi_file = None
    for _ in range(try_times):
        try:
            lock_file.acquire()
            with open(file_name,'rb') as f:
                # this may fail due to unknown bug
                midi_file = io.BytesIO(f.read())
        except BaseException as e:
            try_times -= 1
            time.sleep(1)
            if try_times == 0:
                print('ERROR(READ): ' + file_name +
                      ' ' + str(e) + '\n', end='')
                return None
        finally:
            lock_file.release()
    try:
        with timeout(seconds=600):
            midi_obj = miditoolkit.midi.parser.MidiFile(file=midi_file)
        # check abnormal values in parse result
        assert all(0 <= j.start < 2 ** 31 and 0 <= j.end < 2 **
                   31 for i in midi_obj.instruments for j in i.notes), 'bad note time'
        assert all(0 < j.numerator < 2 ** 31 and 0 < j.denominator < 2 **
                   31 for j in midi_obj.time_signature_changes), 'bad time signature value'
        assert 0 < midi_obj.ticks_per_beat < 2 ** 31, 'bad ticks per beat'
    except BaseException as e:
        print('ERROR(PARSE): ' + file_name + ' ' + str(e) + '\n', end='')
        return None
    midi_notes_count = sum(len(inst.notes) for inst in midi_obj.instruments)
    if midi_notes_count == 0:
        print('ERROR(BLANK): ' + file_name + '\n', end='')
        return None
    e = MIDI_to_encoding(midi_obj)


    ############################################

    if len(e) == 0:
        print('ERROR(BLANK): ' + file_name + '\n', end='')
        return None
    output_str_list = []
    sample_step = max(round(sample_len_max / sample_overlap_rate), 1)
    for p in range(0 - random.randint(0, sample_len_max - 1), len(e), sample_step):
        L = max(p, 0)
        R = min(p + sample_len_max, len(e)) - 1
        bar_index_list = [e[i][0]
                            for i in range(L, R + 1) if e[i][0] is not None]
        bar_index_min = 0
        bar_index_max = 0
        if len(bar_index_list) > 0:
            bar_index_min = min(bar_index_list)
            bar_index_max = max(bar_index_list)
        offset_lower_bound = -bar_index_min
        offset_upper_bound = bar_max - 1 - bar_index_max
        # to make bar index distribute in [0, bar_max)
        bar_index_offset = random.randint(
            offset_lower_bound, offset_upper_bound) if offset_lower_bound <= offset_upper_bound else offset_lower_bound
        e_segment = []
        for i in e[L: R + 1]:
            if i[0] is None or i[0] + bar_index_offset < bar_max:
                e_segment.append(i)
            else:
                break
        tokens_per_note = 8
        output_words = (['<s>'] * tokens_per_note) \
            + [('<{}-{}>'.format(j, k if j > 0 else k + bar_index_offset) if k is not None else '<unk>') for i in e_segment for j, k in enumerate(i)] \
            + (['</s>'] * (tokens_per_note - 1)
                )  # tokens_per_note - 1 for append_eos functionality of binarizer in fairseq
        output_str_list.append(' '.join(output_words))

    # no empty
    if not all(len(i.split()) > tokens_per_note * 2 - 1 for i in output_str_list):
        print('ERROR(ENCODE): ' + file_name + ' ' + str(e) + '\n', end='')
        return False
    try:
        lock_write.acquire()
        writer('l/out.txt', output_str_list)
    except BaseException as e:
        print('ERROR(WRITE): ' + file_name + ' ' + str(e) + '\n', end='')
        return False
    finally:
        lock_write.release()
    print('SUCCESS: ' + file_name + '\n', end='')
    return True

if __name__ == '__main__':
    f = "/Users/arashsadeghiamjadi/Desktop/WORKDIR/muzic/lmd_full/0/"+"0cc576ca78d99cd6a7153dffb453165b.mid"
    F(f)
