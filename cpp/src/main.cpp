// Mirrors bardecode/__main__.py — CLI entry point

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "detector.hpp"
#include "image_io.hpp"
#include "pipeline.hpp"
#include "types.hpp"

namespace {

constexpr char kVersion[] = "0.1.0";

struct Args {
    std::vector<std::string> images;
    double conf_threshold = 0.25;
    double iou_threshold = 0.45;
    int img_size = 640;
    std::string model;
    std::string formats;  // comma-separated whitelist
    bool no_fallback = false;
    bool quiet = false;
    bool verbose = false;
    bool help = false;
    bool version = false;
};

Args parse_args(int argc, char **argv) {
    Args a;
    auto need_value = [&](int &i) -> std::string {
        if (i + 1 >= argc) {
            std::cerr << "error: missing value for " << argv[i] << "\n";
            std::exit(1);
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--conf-threshold") a.conf_threshold = std::stod(need_value(i));
        else if (arg == "--iou-threshold") a.iou_threshold = std::stod(need_value(i));
        else if (arg == "--img-size") a.img_size = std::stoi(need_value(i));
        else if (arg == "--model") a.model = need_value(i);
        else if (arg == "--formats") a.formats = need_value(i);
        else if (arg == "--no-fallback") a.no_fallback = true;
        else if (arg == "-q" || arg == "--quiet") a.quiet = true;
        else if (arg == "-v" || arg == "--verbose") a.verbose = true;
        else if (arg == "-h" || arg == "--help") a.help = true;
        else if (arg == "--version") a.version = true;
        else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "error: unknown option " << arg << "\n";
            std::exit(1);
        } else {
            a.images.push_back(arg);
        }
    }
    return a;
}

void print_help(const char *prog) {
    std::cout
        << "usage: " << prog
        << " [-h] [--conf-threshold CONF_THRESHOLD] [--iou-threshold IOU_THRESHOLD]\n"
           "               [--img-size IMG_SIZE] [--model MODEL] [--formats FORMATS]\n"
           "               [--no-fallback] [-q] [-v] [--version]\n"
           "               images [images ...]\n\n"
           "Detect and decode barcodes from photos.\n";
}

}  // namespace

int main(int argc, char **argv) {
    Args args = parse_args(argc, argv);
    if (args.help) {
        print_help(argv[0]);
        return 0;
    }
    if (args.version) {
        std::cout << "bardecode " << kVersion << "\n";
        return 0;
    }
    if (args.images.empty()) {
        print_help(argv[0]);
        std::cerr << "error: the following arguments are required: images\n";
        return 1;
    }

    std::vector<std::string> formats;
    if (!args.formats.empty()) {
        std::string s = args.formats;
        size_t pos;
        while ((pos = s.find(',')) != std::string::npos) {
            formats.push_back(s.substr(0, pos));
            s.erase(0, pos + 1);
        }
        formats.push_back(s);
    }

    bardecode::Report report;
    report.tool = "bardecode";
    report.version = kVersion;

    int exit_code = 0;
    try {
        bardecode::BarcodeDetector detector(args.model, args.img_size,
                                            args.conf_threshold, args.iou_threshold);
        for (const auto &path : args.images) {
            bardecode::ImageResult ir;
            try {
                ir = bardecode::process_image(path, formats, !args.no_fallback, &detector);
            } catch (const std::exception &e) {
                ir.image = path;
                ir.error = std::string("unexpected: ") + e.what();
            }
            if (args.verbose) {
                std::cerr << "[bardecode] " << path << ": " << ir.barcodes.size()
                          << " decoded, " << ir.undecoded.size()
                          << " undecoded, error="
                          << (ir.error ? ir.error->c_str() : "None") << "\n";
            }
            if (ir.error && *ir.error == "file not found") exit_code = std::max(exit_code, 2);
            else if (ir.error) exit_code = std::max(exit_code, 1);
            report.results.push_back(std::move(ir));
        }
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    std::string out = bardecode::format_report_json(report, args.quiet);
    if (!out.empty()) std::cout << out << "\n";
    return exit_code;
}
