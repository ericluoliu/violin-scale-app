import subprocess
import pathlib
import os.path
from numpy.lib.histograms import _histogram_dispatcher
from werkzeug.utils import secure_filename
from aubio import source, pitch
from flask import Flask, render_template, request
from pydub import AudioSegment
from pydub.utils import make_chunks
import numpy as np
from notes import createNotes
from frequencies import createFrequencies

ALLOWED_EXTENSIONS = {"wav"}
NUMBER_OF_NOTES = 15
PERCENTAGE = 100

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/audio", methods=["GET", "POST"])
def get_audio():
    if request.method == "POST":
        form = request.form
        direction = form["direction"]
        global frequencies
        frequencies = createFrequencies(form["scale"], direction)
        global notes
        notes = createNotes(form["scale"], direction)
        return render_template("audio.html")


@app.route("/result", methods=["GET", "POST"])
def get_result():
    if request.method == "POST":
        f = request.files["audioFile"]
        if allowed_file(f.filename):
            f.save(os.path.join(find_path("uploads"), secure_filename(f.filename)))
            calculate_score(f)
            letterGrade = getLetterGrade(accuracy)
            return render_template(
                "result.html",
                accuracy=accuracy,
                deviations=deviations,
                letterGrade=letterGrade,
                notes=notes,
                zip=zip,
            )
        else:
            return "please upload a .wav file"

    elif request.method == "GET":
        return "please select a file"


def find_path(folder):
    return str(pathlib.Path().resolve()) + "/" + folder


def calculate_score(f):
    inputFrequencies = [None] * NUMBER_OF_NOTES

    # trim silence
    if f.filename.endswith(".wav"):
        f.filename = f.filename[:-4]
    subprocess.run(
        [
            "sox",
            f.filename + ".wav",
            f.filename + "-trimmed.wav",
            "silence",
            "4",
            "0.1",
            "0.1%",
            "reverse",
            "silence",
            "4",
            "0.1",
            "0.1",
            "reverse",
        ],
        cwd=find_path("uploads"),
    )
    trimmed_sound = AudioSegment.from_wav(
        (find_path("uploads") + "/" + f.filename) + "-trimmed.wav"
    )

    # split into chunks
    chunk_size = len(trimmed_sound) / NUMBER_OF_NOTES
    chunks = make_chunks(trimmed_sound, chunk_size)

    # check pitch of each chunk
    for i, chunk in enumerate(chunks):

        samplerate = 44100

        chunk_name = "chunk{0}.wav".format(i)
        chunk.export(find_path("uploads") + "/" + chunk_name, format="wav")

        win_s = 4096
        hop_s = 512

        s = source(find_path("uploads") + "/" + chunk_name, samplerate, hop_s)
        samplerate = s.samplerate

        tolerance = 0.8

        pitch_o = pitch("mcomb", win_s, hop_s, samplerate)
        pitch_o.set_unit("Hz")
        pitch_o.set_tolerance(tolerance)

        pitches = []
        confidences = []

        total_frames = 0
        while True:
            samples, read = s()
            note = pitch_o(samples)[0]
            pitches += [note]
            confidence = pitch_o.get_confidence()
            confidences += [confidence]
            total_frames += read
            if read < hop_s:
                break

        inputFrequencies[i] = np.array(pitches).mean()

    # calculate score based off percentage error calculation and calculate frequency deviations
    global accuracy
    global deviations
    deviations = [None] * NUMBER_OF_NOTES
    percent_error = 0
    index = 0
    for i, j in zip(frequencies, inputFrequencies):
        percent_error += abs(i - j) / i
        boolean_diff = "higher "
        if i - j < 0:
            boolean_diff = "lower "
        deviations[index] = boolean_diff + str(round(abs(i - j), 2))
        index += 1

    percent_error *= PERCENTAGE / NUMBER_OF_NOTES
    accuracy = 100 - percent_error
    accuracy = str(round(accuracy, 2))


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def getLetterGrade(score):
    convertedScore = float(score)
    if convertedScore > 99:
        return "S"
    if convertedScore > 97:
        return "A"
    if convertedScore > 94:
        return "B"
    if convertedScore > 85:
        return "C"
    if convertedScore > 60:
        return "D"
    return "F"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
