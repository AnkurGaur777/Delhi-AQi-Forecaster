import argparse
import logging

from aqi import clean, fetch


def main() -> None:
    parser = argparse.ArgumentParser(description="Data steps for the Delhi AQI forecaster")
    parser.add_argument("step", choices=["download", "clean", "refresh"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.step == "download":
        summary = fetch.download_history()
        fetch.refresh_recent()
        logging.info("download finished: %s", summary)
    elif args.step == "clean":
        clean.clean_all()
    else:
        fetch.refresh_recent()


if __name__ == "__main__":
    main()
