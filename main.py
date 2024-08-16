import os
import pretty_midi
import json
import subprocess

def note2midi(f_note, f_2note, f_midi):
    with open(f_note, 'r') as f:
        a_note = json.load(f)

    with open(f_2note, 'r') as f:
        a_2note = json.load(f)

    a_note.extend(a_2note)
    a_note = [dict(t) for t in {tuple(d.items()) for d in a_note}]

    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    for note in a_note:
        instrument.notes.append(pretty_midi.Note(velocity=note['velocity'], pitch=note['pitch'], start=note['onset'], end=note['offset']))
    midi.instruments.append(instrument)
    midi.write(f_midi)

    os.remove(f_note)
    os.remove(f_2note)


if __name__ == '__main__':
    file_name = "test.wav"
    subprocess.run(["python3", "evaluation/m_inference.py", "-f_wav", file_name, "-d_wav", "input", "-f_config", "corpus/config.json", "-d_fe", "corpus/feature", "-d_cp", "checkpoint/MAESTRO-V3", "-m", "best_model.pkl", "-d_mpe", "output", "-d_note", "output", "-calc_transcript", "-calc_feature", "-mode", "combination"])
    note2midi(f'output/{file_name}_1st.json', f'output/{file_name}_2nd.json', f'output/{file_name.split("/")[0]}.midi')