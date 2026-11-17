"""Run the same audio-file prediction as the Qt application."""
import argparse
import json

from threadpoolctl import threadpool_limits

from qt_gui.audio import load_preview
from ser.inference import MODELS, Predictor, demo_run, final_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("--model", choices=MODELS, default="svm")
    parser.add_argument("--split", choices=("speaker", "random"), default="speaker")
    parser.add_argument("--noise", choices=("clean", "white", "background"), default="clean")
    parser.add_argument("--snr", type=int, choices=(0, 10, 20), default=10)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        preview = load_preview(args.audio)
        predictor = Predictor(demo_run(final_runs(), args.model, args.split))
        if predictor.name == "cnn":
            predictor.torch.set_num_threads(2)
        print(json.dumps(predictor.compare_waveform(preview.waveform, preview.sample_rate, args.noise, args.snr), indent=2))


if __name__ == "__main__":
    main()
