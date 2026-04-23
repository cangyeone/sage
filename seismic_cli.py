#!/usr/bin/env python3
"""
SeismicX CLI - Command Line Interface for Seismic Analysis Skills

Unified command-line interface for:
- Phase picking (seismic-phase-picker)
- Phase association (seismic-phase-associator)
- Polarity analysis (seismic-polarity-analyzer)

Usage:
    python seismic_cli.py pick [options]
    python seismic_cli.py associate [options]
    python seismic_cli.py polarity [options]
"""

import argparse
import sys
import os
from pathlib import Path

# Import config manager
try:
    from config_manager import get_config_manager
except ImportError:
    # If config_manager not available, create a dummy one
    class DummyConfigManager:
        def is_first_run(self):
            return False
        def interactive_setup(self):
            print("Config manager not available")
        def get_llm_config(self):
            return {}
    def get_config_manager():
        return DummyConfigManager()


def setup_llm_parser(subparsers):
    """Setup LLM configuration subcommand"""
    parser = subparsers.add_parser(
        'llm',
        help='Configure LLM model settings',
        description='Manage LLM provider and model configuration'
    )

    subparsers_llm = parser.add_subparsers(dest='llm_command', help='LLM commands')

    # Setup command
    setup_parser = subparsers_llm.add_parser('setup', help='Interactive setup wizard')

    # Show command
    show_parser = subparsers_llm.add_parser('show', help='Show current configuration')

    # Set provider command
    set_provider = subparsers_llm.add_parser('set-provider', help='Set LLM provider')
    set_provider.add_argument(
        'provider',
        choices=['ollama', 'openai', 'anthropic', 'azure', 'custom'],
        help='LLM provider name'
    )

    # Set model command
    set_model = subparsers_llm.add_parser('set-model', help='Set LLM model')
    set_model.add_argument('model', help='Model name')

    # List models command
    list_models = subparsers_llm.add_parser('list-models', help='List available Ollama models')

    return parser


def handle_llm_command(args):
    """Handle LLM configuration commands"""
    config = get_config_manager()

    if args.llm_command is None:
        # Default to interactive setup
        config.interactive_setup()
    elif args.llm_command == 'setup':
        config.interactive_setup()
    elif args.llm_command == 'show':
        llm_config = config.get_llm_config()
        print("Current LLM Configuration:")
        print("-" * 40)
        for key, value in llm_config.items():
            if key == 'api_key' and value:
                print(f"{key}: {'*' * 8} (hidden)")
            else:
                print(f"{key}: {value}")
    elif args.llm_command == 'set-provider':
        config.set_llm_provider(args.provider)
        print(f"✓ LLM provider set to: {args.provider}")
    elif args.llm_command == 'set-model':
        config.set_llm_model(args.model)
        print(f"✓ LLM model set to: {args.model}")
    elif args.llm_command == 'list-models':
        models = config.get_ollama_models()
        if models:
            print("Available Ollama models:")
            for model in models:
                print(f"  - {model}")
        else:
            print("No Ollama models found. Make sure Ollama is running.")
            print("Run 'ollama list' to check.")


def setup_pick_parser(subparsers):
    """Setup phase picking subcommand"""
    parser = subparsers.add_parser(
        'pick',
        help='Detect seismic phases from waveform data',
        description='Run deep learning models to detect Pg, Sg, Pn, Sn phases from continuous waveforms'
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input directory containing waveform files (mseed/sac format)'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output filename prefix (generates .txt, .log, .err files)'
    )
    parser.add_argument(
        '-m', '--model',
        default='pnsn/pickers/pnsn.v3.jit',
        help='Model file path (.jit or .onnx). Default: pnsn/pickers/pnsn.v3.jit'
    )
    parser.add_argument(
        '-d', '--device',
        default='cpu',
        choices=['cpu', 'cuda', 'cuda:0', 'cuda:1', 'mps'],
        help='Device for inference. Default: cpu'
    )
    parser.add_argument(
        '--config',
        default='pnsn/config/picker.py',
        help='Configuration file path. Default: pnsn/config/picker.py'
    )
    parser.add_argument(
        '--prob-thresh',
        type=float,
        default=None,
        help='Confidence threshold (0-1). Overrides config file'
    )
    parser.add_argument(
        '--nms-window',
        type=int,
        default=None,
        help='NMS window in samples. Overrides config file'
    )
    parser.add_argument(
        '--enable-polarity',
        action='store_true',
        help='Enable first-motion polarity detection'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate waveform plots with picks'
    )

    return parser


def setup_associate_parser(subparsers):
    """Setup phase association subcommand"""
    parser = subparsers.add_parser(
        'associate',
        help='Associate phase picks into earthquake events',
        description='Group phase picks from multiple stations into coherent earthquake events'
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Phase picking result file (.txt from picker)'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output event file name'
    )
    parser.add_argument(
        '-s', '--station',
        required=True,
        help='Station file with coordinates (format: NET STA LOC lon lat elev)'
    )
    parser.add_argument(
        '-m', '--method',
        default='fastlink',
        choices=['fastlink', 'real', 'gamma'],
        help='Association method. Default: fastlink'
    )
    parser.add_argument(
        '-d', '--device',
        default='cpu',
        choices=['cpu', 'cuda', 'cuda:0', 'cuda:1', 'mps'],
        help='Device for neural network (FastLink only). Default: cpu'
    )
    parser.add_argument(
        '--min-phases',
        type=int,
        default=None,
        help='Minimum total phases per event'
    )
    parser.add_argument(
        '--min-p',
        type=int,
        default=None,
        help='Minimum P phases per event'
    )
    parser.add_argument(
        '--min-s',
        type=int,
        default=None,
        help='Minimum S phases per event'
    )

    return parser


def setup_polarity_parser(subparsers):
    """Setup polarity analysis subcommand"""
    parser = subparsers.add_parser(
        'polarity',
        help='Analyze first-motion polarity of P-waves',
        description='Detect upward/downward initial motion of P-waves using deep learning'
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Phase picking result file with picks'
    )
    parser.add_argument(
        '-w', '--waveform-dir',
        required=True,
        help='Directory containing waveform files'
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output file for polarity results (default: update input file)'
    )
    parser.add_argument(
        '--model',
        default='pnsn/pickers/polar.onnx',
        help='Polarity model file. Default: pnsn/pickers/polar.onnx'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.5,
        help='Minimum confidence threshold. Default: 0.5'
    )
    parser.add_argument(
        '--phase',
        default='Pg',
        choices=['Pg', 'Pn', 'P'],
        help='Phase type to analyze. Default: Pg'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate polarity visualization plots'
    )
    parser.add_argument(
        '--plot-dir',
        default='output/plots',
        help='Directory for plot output. Default: output/plots'
    )

    return parser


def run_picking(args):
    """Execute phase picking workflow"""
    print("=" * 80)
    print("SeismicX Phase Picker")
    print("=" * 80)

    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Error: Input directory not found: {args.input}")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    # Update config if needed
    if args.prob_thresh is not None or args.nms_window is not None or args.enable_polarity or args.plot:
        update_picker_config(
            args.config,
            prob=args.prob_thresh,
            nmslen=args.nms_window,
            polar=args.enable_polarity,
            ifplot=args.plot
        )

    # Build command
    cmd = f"python pnsn/picker.py -i {args.input} -o {args.output} -m {args.model} -d {args.device}"

    print(f"\nRunning command:")
    print(f"  {cmd}\n")

    # Execute
    import subprocess
    result = subprocess.run(cmd, shell=True)

    if result.returncode == 0:
        print(f"\n✓ Phase picking completed successfully!")
        print(f"  Results: {args.output}.txt")
        print(f"  Log: {args.output}.log")
        print(f"  Errors: {args.output}.err")
    else:
        print(f"\n✗ Phase picking failed with code {result.returncode}")
        sys.exit(1)


def run_association(args):
    """Execute phase association workflow"""
    print("=" * 80)
    print(f"SeismicX Phase Associator ({args.method.upper()})")
    print("=" * 80)

    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    if not os.path.exists(args.station):
        print(f"Error: Station file not found: {args.station}")
        sys.exit(1)

    # Select association script
    method_scripts = {
        'fastlink': 'pnsn/fastlinker.py',
        'real': 'pnsn/reallinker.py',
        'gamma': 'pnsn/gammalink.py'
    }

    script = method_scripts[args.method]
    if not os.path.exists(script):
        print(f"Error: Association script not found: {script}")
        sys.exit(1)

    # Build command
    cmd = f"python {script} -i {args.input} -o {args.output} -s {args.station}"
    if args.method == 'fastlink':
        cmd += f" -d {args.device}"

    print(f"\nRunning command:")
    print(f"  {cmd}\n")

    # Execute
    import subprocess
    result = subprocess.run(cmd, shell=True)

    if result.returncode == 0:
        print(f"\n✓ Phase association completed successfully!")
        print(f"  Events: {args.output}")
    else:
        print(f"\n✗ Phase association failed with code {result.returncode}")
        sys.exit(1)


def run_polarity(args):
    """Execute polarity analysis workflow"""
    print("=" * 80)
    print("SeismicX Polarity Analyzer")
    print("=" * 80)

    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    if not os.path.exists(args.waveform_dir):
        print(f"Error: Waveform directory not found: {args.waveform_dir}")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"Error: Polarity model not found: {args.model}")
        sys.exit(1)

    # Create output directory if needed
    if args.plot and not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir, exist_ok=True)

    print(f"\nAnalyzing polarity for {args.phase} phases...")
    print(f"  Input: {args.input}")
    print(f"  Waveforms: {args.waveform_dir}")
    print(f"  Model: {args.model}")
    print(f"  Min confidence: {args.min_confidence}")

    # Run polarity analysis
    try:
        import obspy
        import numpy as np
        import onnxruntime as ort

        # Load model
        sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])

        # Read picks
        picks_by_station = {}
        current_file = None

        with open(args.input, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    current_file = line.strip()[1:]
                    continue

                parts = line.strip().split(',')
                if len(parts) < 7:
                    continue

                phase_name = parts[0]
                station = parts[6]

                if phase_name != args.phase:
                    continue

                if station not in picks_by_station:
                    picks_by_station[station] = []

                picks_by_station[station].append({
                    'file': current_file,
                    'rel_time': float(parts[1]),
                    'confidence': float(parts[2]),
                    'abs_time': parts[3],
                    'line': line.strip()
                })

        # Process each station
        results = []
        for station, picks in picks_by_station.items():
            print(f"\nProcessing station: {station} ({len(picks)} picks)")

            # Find waveform file
            sample_pick = picks[0]
            waveform_path = os.path.join(args.waveform_dir,
                                        os.path.basename(sample_pick['file']))

            if not os.path.exists(waveform_path):
                print(f"  Warning: Waveform not found: {waveform_path}")
                continue

            try:
                st = obspy.read(waveform_path)
                tr_z = st.select(component='Z')

                if len(tr_z) == 0:
                    print(f"  Warning: No vertical component found")
                    continue

                tr_z = tr_z[0]
                data = tr_z.data.astype(np.float32)

                for pick in picks:
                    pick_idx = int(pick['rel_time'] * tr_z.stats.sampling_rate)

                    # Check bounds
                    if pick_idx < 512 or pick_idx > len(data) - 512:
                        results.append({
                            'station': station,
                            'time': pick['abs_time'],
                            'polarity': 'N',
                            'prob': 0.0,
                            'reason': 'out_of_bounds'
                        })
                        continue

                    # Extract window
                    pdata = data[pick_idx-512:pick_idx+512]

                    # Ensure correct length
                    if len(pdata) < 1024:
                        pdata = np.pad(pdata, (0, 1024-len(pdata)))
                    elif len(pdata) > 1024:
                        pdata = pdata[:1024]

                    # Run inference
                    prob, = sess.run(["prob"], {"wave": pdata})
                    polarity_id = np.argmax(prob)
                    polarity = "U" if polarity_id == 0 else "D"
                    confidence = np.max(prob)

                    # Apply threshold
                    if confidence < args.min_confidence:
                        polarity = "N"

                    results.append({
                        'station': station,
                        'time': pick['abs_time'],
                        'polarity': polarity,
                        'prob': float(confidence),
                        'reason': 'ok'
                    })

                    print(f"  {pick['abs_time']}: {polarity} ({confidence:.3f})")

            except Exception as e:
                print(f"  Error processing {station}: {e}")
                continue

        # Write results
        output_file = args.output if args.output else args.input.replace('.txt', '_polarity.txt')

        with open(output_file, 'w') as f:
            f.write("# Station, Time, Polarity, Confidence\n")
            for r in results:
                f.write(f"{r['station']},{r['time']},{r['polarity']},{r['prob']:.3f}\n")

        print(f"\n✓ Polarity analysis completed!")
        print(f"  Results: {output_file}")
        print(f"  Total analyzed: {len(results)}")
        print(f"  Upward (U): {sum(1 for r in results if r['polarity'] == 'U')}")
        print(f"  Downward (D): {sum(1 for r in results if r['polarity'] == 'D')}")
        print(f"  Uncertain (N): {sum(1 for r in results if r['polarity'] == 'N')}")

    except ImportError as e:
        print(f"Error: Required package not installed: {e}")
        print("Install with: pip install obspy numpy onnxruntime")
        sys.exit(1)


def update_picker_config(config_path, prob=None, nmslen=None, polar=None, ifplot=None):
    """Update picker configuration file"""
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        lines = f.readlines()

    modified = False
    new_lines = []

    for line in lines:
        if prob is not None and line.strip().startswith('prob ='):
            new_lines.append(f"    prob = {prob}                   # Confidence threshold\n")
            modified = True
        elif nmslen is not None and line.strip().startswith('nmslen ='):
            new_lines.append(f"    nmslen = {nmslen}                # NMS window in samples\n")
            modified = True
        elif polar is not None and 'polar =' in line and '#' in line:
            new_lines.append(f"    polar = {'True' if polar else 'False'} # Enable polarity detection\n")
            modified = True
        elif ifplot is not None and line.strip().startswith('ifplot ='):
            new_lines.append(f"    ifplot = {'True' if ifplot else 'False'}               # Generate plots\n")
            modified = True
        else:
            new_lines.append(line)

    if modified:
        with open(config_path, 'w') as f:
            f.writelines(new_lines)
        print(f"Updated config: {config_path}")


def setup_stats_parser(subparsers):
    """Setup seismological statistics subcommand."""
    parser = subparsers.add_parser(
        'stats',
        help='Compute seismological statistics: b-value, Mc, G-R relation and distribution plots',
        description=(
            'Load a SAGE picks file or an earthquake catalog (CSV/JSON) and compute '
            'the Gutenberg-Richter b-value, completeness magnitude Mc, and optional figures.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect newest picks file in results/ and show stats
  python seismic_cli.py stats -i results/

  # Catalog CSV with known Mc, generate all plots
  python seismic_cli.py stats -i catalog.csv --mc 2.0 --plot

  # Specify output prefix and b-value method
  python seismic_cli.py stats -i catalog.json -o results/myrun --method lsq --plot

  # Only compute Mc (no b-value if no magnitudes available)
  python seismic_cli.py stats -i results/sage_picks_20260422_194552.txt
        """
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help=(
            'Input data path. Can be:\n'
            '  - A SAGE picks .txt file (sage_picks_*.txt)\n'
            '  - A directory containing sage_picks_*.txt files\n'
            '  - An earthquake catalog file (.csv or .json) with magnitude column'
        )
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help=(
            'Output file prefix for figures and text report. '
            'Default: results/stats_YYYYMMDD_HHMMSS'
        )
    )
    parser.add_argument(
        '--mc',
        type=float,
        default=None,
        help='Completeness magnitude Mc. Estimated automatically if not given.'
    )
    parser.add_argument(
        '--mc-method',
        default='maxcurvature',
        choices=['maxcurvature', 'gof'],
        help='Mc estimation method when --mc is not supplied. Default: maxcurvature'
    )
    parser.add_argument(
        '--method',
        default='mle',
        choices=['mle', 'lsq'],
        help='b-value estimation method: mle (maximum likelihood) or lsq (least squares). Default: mle'
    )
    parser.add_argument(
        '--mag-bin',
        type=float,
        default=0.1,
        help='Magnitude bin width for FMD. Default: 0.1'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate analysis figures (G-R plot, temporal and spatial distribution)'
    )
    parser.add_argument(
        '--no-spatial',
        action='store_true',
        help='Skip spatial distribution plot even when --plot is given'
    )
    parser.add_argument(
        '--no-temporal',
        action='store_true',
        help='Skip temporal distribution plot even when --plot is given'
    )
    return parser


def run_stats(args):
    """Execute seismological statistics analysis."""
    import sys
    from pathlib import Path as _Path
    from datetime import datetime as _dt

    # Ensure project root is on path
    project_root = str(_Path(__file__).parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from seismo_stats.catalog_loader import load_picks_txt, load_catalog_file
        from seismo_stats.bvalue import calc_bvalue_mle, calc_bvalue_lsq
        from seismo_stats.plotting import plot_gr, plot_temporal, plot_spatial
    except ImportError as e:
        print(f"✗ 导入 seismo_stats 模块失败: {e}")
        print("  请确认 seismo_stats/ 目录在项目根目录下。")
        sys.exit(1)

    input_path = _Path(args.input)
    print(f"加载数据: {input_path}")

    # --- Load data ---
    try:
        if input_path.is_dir() or (input_path.is_file() and 'sage_picks' in input_path.name):
            catalog = load_picks_txt(str(input_path))
            print(f"  → 识别为 SAGE 震相拾取文件")
        else:
            catalog = load_catalog_file(str(input_path))
            print(f"  → 识别为地震目录文件")
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        sys.exit(1)

    print(catalog.summary())
    print()

    # --- Output prefix ---
    if args.output:
        output_prefix = args.output
    else:
        results_dir = _Path(project_root) / 'results'
        results_dir.mkdir(exist_ok=True)
        ts = _dt.now().strftime('%Y%m%d_%H%M%S')
        output_prefix = str(results_dir / f'stats_{ts}')

    # --- b-value ---
    result_bv = None
    if catalog.has_magnitudes:
        try:
            if args.method == 'lsq':
                result_bv = calc_bvalue_lsq(
                    catalog.magnitudes,
                    mc=args.mc,
                    mag_bin=args.mag_bin,
                    mc_method=args.mc_method,
                )
            else:
                result_bv = calc_bvalue_mle(
                    catalog.magnitudes,
                    mc=args.mc,
                    mag_bin=args.mag_bin,
                    mc_method=args.mc_method,
                )
            print(result_bv.summary())
        except Exception as e:
            print(f"⚠  b值计算失败: {e}")
    else:
        print("⚠  数据中无震级信息，跳过 b 值计算。")
        print("   若需计算 b 值，请提供含 magnitude/mag/ML 列的地震目录文件（CSV/JSON）。")

    # --- Plots ---
    if args.plot:
        print()
        print("生成图像...")

        if result_bv is not None:
            try:
                p = plot_gr(result_bv, output_prefix + '_gr.png')
                print(f"  ✓ G-R 频率震级关系图: {p}")
            except Exception as e:
                print(f"  ✗ G-R 图失败: {e}")

        if not args.no_temporal and catalog.times:
            try:
                p = plot_temporal(catalog, output_prefix + '_temporal.png')
                print(f"  ✓ 时间分布图: {p}")
            except Exception as e:
                print(f"  ✗ 时间分布图失败: {e}")

        if not args.no_spatial:
            if catalog.has_locations:
                try:
                    p = plot_spatial(catalog, output_prefix + '_spatial.png')
                    print(f"  ✓ 空间分布图: {p}")
                except Exception as e:
                    print(f"  ✗ 空间分布图失败: {e}")
            else:
                print("  ⚠  数据无经纬度，跳过空间分布图")

    print()
    print(f"✓ 统计分析完成，输出前缀: {output_prefix}")


def setup_run_parser(subparsers):
    """Setup LLM code generation subcommand."""
    parser = subparsers.add_parser(
        'run',
        help='用自然语言描述地震学任务，LLM 自动生成并执行 Python 代码',
        description='LLM 驱动的地震学代码生成与执行引擎',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python seismic_cli.py run "对 /data/wave.mseed 做 1-10Hz 带通滤波并画图"
  python seismic_cli.py run "计算震源参数，震中距 50km" -d /data/waves/
  python seismic_cli.py run "画走时曲线，距离 0-30°，深度 10km，iasp91 模型"
  python seismic_cli.py run "计算 ML 震级" -d /data/ --timeout 120
        """
    )
    parser.add_argument('request', help='自然语言任务描述')
    parser.add_argument('-d', '--data', default=None,
                        help='数据文件或目录路径（提示 LLM 使用）')
    parser.add_argument('--timeout', type=int, default=90,
                        help='代码执行超时（秒），默认 90')
    parser.add_argument('--retries', type=int, default=2,
                        help='执行失败时最大重试次数，默认 2')
    parser.add_argument('--show-code', action='store_true',
                        help='打印生成的 Python 代码')
    return parser


def run_code(args):
    """Execute LLM code generation task."""
    import sys
    from pathlib import Path as _Path

    project_root = str(_Path(__file__).parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from seismo_code.code_engine import get_code_engine
    except ImportError as e:
        print(f"✗ 导入 seismo_code 模块失败: {e}")
        sys.exit(1)

    engine = get_code_engine()

    if not engine.is_llm_available():
        print("✗ LLM 服务不可用。")
        print("  - 若使用 Ollama：请先运行 `ollama serve`")
        print("  - 若使用 API：请运行 `python seismic_cli.py llm setup` 配置")
        sys.exit(1)

    print(f"任务: {args.request}")
    if args.data:
        print(f"数据: {args.data}")
    print("LLM 生成代码中...\n")

    result = engine.run(
        user_request=args.request,
        data_hint=args.data,
        max_retries=args.retries,
        timeout=args.timeout,
    )

    if args.show_code and result.code:
        print("=" * 60)
        print("生成的代码:")
        print("=" * 60)
        print(result.code)
        print("=" * 60)
        print()

    if result.stdout.strip():
        print("输出:")
        for line in result.stdout.splitlines():
            if not line.startswith("[FIGURE]"):
                print(f"  {line}")
        print()

    if result.figures:
        print(f"生成图像 ({len(result.figures)} 张):")
        for f in result.figures:
            print(f"  • {f}")

    if result.success:
        print("\n✓ 执行成功")
    else:
        print(f"\n✗ 执行失败")
        if result.exec_result and result.exec_result.error:
            print(f"  错误: {result.exec_result.error}")


def setup_tool_parser(subparsers):
    """Setup external tool management subcommand."""
    parser = subparsers.add_parser(
        'tool',
        help='管理地震学外部工具（HypoDD、VELEST、NonLinLoc 等）',
        description='外部地震学工具注册、文档解析和输入文件生成',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
子命令:
  list              列出所有可用工具
  info TOOL         显示工具详细信息（如 info hypodd）
  parse FILE        解析文档文件，自动注册工具接口
  gen TOOL          为指定工具生成输入文件（需提供数据）

示例:
  python seismic_cli.py tool list
  python seismic_cli.py tool info hypodd
  python seismic_cli.py tool parse /path/to/readme.txt
  python seismic_cli.py tool gen hypodd --data-dir /data/ --output-dir /work/run/
        """
    )
    sub = parser.add_subparsers(dest='tool_command')

    # list
    sub.add_parser('list', help='列出所有可用工具')

    # info
    info_p = sub.add_parser('info', help='显示工具详细信息')
    info_p.add_argument('tool_name', help='工具名称（如 hypodd, velest, nonlinloc）')

    # parse
    parse_p = sub.add_parser('parse', help='解析工具文档文件并注册')
    parse_p.add_argument('doc_file', help='文档文件路径（.txt/.md/.rst/.pdf）')

    # gen
    gen_p = sub.add_parser('gen', help='为工具生成输入文件')
    gen_p.add_argument('tool_name', help='工具名称')
    gen_p.add_argument('--data-dir', '-d', default=None,
                       help='数据目录（含震相文件、台站文件等）')
    gen_p.add_argument('--picks-file', default=None, help='震相拾取文件路径')
    gen_p.add_argument('--station-file', default=None, help='台站坐标文件路径')
    gen_p.add_argument('--output-dir', '-o', required=True, help='输出目录（存放生成的输入文件）')

    return parser


def handle_tool_command(args):
    """Handle external tool management commands."""
    import sys
    from pathlib import Path as _Path

    project_root = str(_Path(__file__).parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from seismo_tools.tool_registry import list_tools, get_tool, generate_input_files
    except ImportError as e:
        print(f"✗ 导入 seismo_tools 模块失败: {e}")
        sys.exit(1)

    cmd = getattr(args, 'tool_command', None)

    if cmd is None or cmd == 'list':
        tools = list_tools()
        print("可用地震学工具:")
        print()
        builtin = ['hypodd', 'velest', 'nonlinloc', 'hypoinverse', 'focmec']
        print("  内置工具（含完整接口模板）:")
        for t in builtin:
            p = get_tool(t)
            if p:
                print(f"    {t:15s} — {p.get('description', '')[:60]}")
        others = [t for t in tools if t not in builtin]
        if others:
            print()
            print("  用户注册工具:")
            for t in others:
                print(f"    {t}")
        print()
        print("查看详情: python seismic_cli.py tool info <工具名>")

    elif cmd == 'info':
        profile = get_tool(args.tool_name)
        if profile is None:
            print(f"✗ 未找到工具: {args.tool_name}")
            print(f"  可用工具: {list_tools()}")
            sys.exit(1)
        print(f"工具: {profile['name']}")
        print(f"程序: {profile['executable']}")
        print(f"描述: {profile['description']}")
        print(f"输入文件: {', '.join(profile['input_files'])}")
        print(f"输出文件: {', '.join(profile['output_files'])}")
        print()
        if profile.get('input_format'):
            print("输入格式说明:")
            print(profile['input_format'])
        if profile.get('run_command'):
            print(f"调用命令: {profile['run_command']}")
        if profile.get('notes'):
            print(f"注意事项: {profile['notes']}")

    elif cmd == 'parse':
        try:
            from seismo_code.doc_parser import DocParser
        except ImportError as e:
            print(f"✗ 导入文档解析模块失败: {e}")
            sys.exit(1)

        print(f"解析文档: {args.doc_file}")
        parser = DocParser()
        try:
            profile = parser.parse_file(args.doc_file)
            saved = profile.save()
            print(f"✓ 已注册工具: {profile.name}")
            print(f"  保存路径: {saved}")
            print()
            print(profile.summary())
        except Exception as e:
            print(f"✗ 解析失败: {e}")
            sys.exit(1)

    elif cmd == 'gen':
        data_info = {}
        if args.data_dir:
            data_info['data_directory'] = args.data_dir
        if args.picks_file:
            data_info['picks_file'] = args.picks_file
        if args.station_file:
            data_info['station_file'] = args.station_file

        if not data_info:
            print("✗ 请指定数据信息（--data-dir / --picks-file / --station-file）")
            sys.exit(1)

        print(f"为 {args.tool_name} 生成输入文件...")
        try:
            written = generate_input_files(args.tool_name, data_info, args.output_dir)
            print(f"✓ 已生成 {len(written)} 个输入文件:")
            for fname, fpath in written.items():
                print(f"  • {fname}: {fpath}")
            profile = get_tool(args.tool_name)
            if profile:
                print(f"\n运行命令（在 {args.output_dir} 目录下执行）:")
                print(f"  cd {args.output_dir} && {profile.get('run_command', args.tool_name)}")
        except Exception as e:
            print(f"✗ 生成失败: {e}")
            sys.exit(1)

    else:
        print(f"未知子命令: {cmd}")
        print("可用子命令: list, info, parse, gen")


def setup_backend_parser(subparsers):
    """Setup backend management subcommand."""
    parser = subparsers.add_parser(
        'backend',
        help='管理 LLM 后端（Ollama / vLLM / 在线 API）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
子命令示例:
  # 查看所有后端状态
  sage backend status

  # 切换到 Ollama（自动选模型）
  sage backend use ollama
  sage backend use ollama --model qwen2.5:14b

  # 安装并启动 vLLM
  sage backend install vllm
  sage backend use vllm --model ~/.seismicx/models/Qwen2.5-7B-Instruct
  sage backend start-vllm
  sage backend stop-vllm

  # 查看/下载本地模型
  sage backend models
  sage backend download-guide

  # 切换到在线 API（DeepSeek 等）
  sage backend use online --provider deepseek --api-key sk-xxx
  sage backend use online --provider siliconflow --api-key sk-xxx --model Qwen/Qwen2.5-7B-Instruct

  # 拉取 Ollama 模型
  sage backend pull qwen2.5:7b

  # 交互式向导
  sage backend setup
        """
    )
    sub = parser.add_subparsers(dest='backend_cmd', metavar='SUBCMD')

    # status
    sub.add_parser('status', help='显示所有后端状态')

    # setup (interactive wizard)
    sub.add_parser('setup', help='交互式配置向导')

    # use
    use_p = sub.add_parser('use', help='切换激活后端')
    use_p.add_argument('backend_type', choices=['ollama', 'vllm', 'online'],
                       help='后端类型')
    use_p.add_argument('--model', default=None,
                       help='Ollama: 模型标签 | vLLM: 本地路径 | online: 模型名')
    use_p.add_argument('--api-base', default=None, help='API base URL（可选）')
    use_p.add_argument('--provider', default='deepseek',
                       help='在线 API 服务商（openai/deepseek/siliconflow/moonshot/zhipu/dashscope/custom）')
    use_p.add_argument('--api-key', default=None, help='在线 API Key')
    use_p.add_argument('--port', type=int, default=None, help='vLLM 端口（默认 8001）')

    # install
    inst_p = sub.add_parser('install', help='安装后端')
    inst_p.add_argument('target', choices=['vllm'], help='要安装的后端')
    inst_p.add_argument('--cpu', action='store_true', help='安装 CPU-only 版本')

    # start-vllm
    start_p = sub.add_parser('start-vllm', help='启动 vLLM 服务')
    start_p.add_argument('--model', default=None, help='模型路径（默认使用配置中的路径）')
    start_p.add_argument('--port', type=int, default=None, help='端口')
    start_p.add_argument('--gpu-mem', type=float, default=0.9, help='GPU 显存占比 (0-1)')

    # stop-vllm
    sub.add_parser('stop-vllm', help='停止 vLLM 服务')

    # models
    sub.add_parser('models', help='列出本地模型目录')

    # download-guide
    dg_p = sub.add_parser('download-guide', help='显示模型下载教程')
    dg_p.add_argument('--model', type=int, default=None,
                      help='推荐模型序号（不填显示通用教程）')

    # pull (Ollama)
    pull_p = sub.add_parser('pull', help='拉取 Ollama 模型')
    pull_p.add_argument('model_tag', help='模型标签，如 qwen2.5:7b')

    # auto
    sub.add_parser('auto', help='自动探测并选择最佳可用后端')

    return parser


def handle_backend_command(args):
    """Handle backend subcommand."""
    import sys as _sys
    from pathlib import Path as _Path
    _proj = str(_Path(__file__).parent)
    if _proj not in _sys.path:
        _sys.path.insert(0, _proj)

    try:
        from backend_manager import BackendManager, RECOMMENDED_LOCAL_MODELS, MODEL_DIR
    except ImportError as e:
        print(f"✗ 无法导入 backend_manager: {e}")
        return

    mgr = BackendManager()
    cmd = getattr(args, 'backend_cmd', None)

    if cmd is None or cmd == 'status':
        mgr.print_status()

    elif cmd == 'setup':
        mgr.interactive_setup()

    elif cmd == 'auto':
        mgr.auto_select()
        mgr.print_status()

    elif cmd == 'use':
        bt = args.backend_type
        if bt == 'ollama':
            mgr.use_ollama(model=args.model, api_base=args.api_base)
            model = args.model or mgr._config.get('ollama', {}).get('model', 'qwen2.5:7b')
            print(f"✓ 切换到 Ollama，模型: {model}")
        elif bt == 'vllm':
            if not args.model:
                # 列出已有本地模型让用户选
                lm = mgr.list_local_models()
                if lm:
                    print("本地模型:")
                    for i, p in enumerate(lm, 1):
                        print(f"  {i}. {p}")
                    raw = input("输入序号或路径: ").strip()
                    try:
                        idx = int(raw) - 1
                        model_path = str(lm[idx])
                    except (ValueError, IndexError):
                        model_path = raw
                else:
                    print(f"✗ 请用 --model 指定路径，或先用 `sage backend download-guide` 下载模型")
                    return
            else:
                model_path = args.model
            mgr.use_vllm(model_path, port=args.port)
            print(f"✓ 切换到 vLLM，模型: {model_path}")
            print(f"  运行 `sage backend start-vllm` 启动服务")
        elif bt == 'online':
            api_key = args.api_key or ''
            if not api_key:
                api_key = input(f"输入 {args.provider} API Key: ").strip()
            mgr.use_online(args.provider, api_key, model=args.model, api_base=args.api_base)
            print(f"✓ 切换到在线 API: {args.provider}")

    elif cmd == 'install':
        if args.target == 'vllm':
            mgr.install_vllm(cpu_only=args.cpu)

    elif cmd == 'start-vllm':
        vcfg = mgr._config.get('vllm', {})
        model_path = args.model or vcfg.get('model_path', '')
        if not model_path:
            print("✗ 未指定模型路径，请用 --model 参数或先运行 `sage backend use vllm --model <路径>`")
            return
        mgr.start_vllm(model_path, port=args.port, gpu_memory_fraction=args.gpu_mem)

    elif cmd == 'stop-vllm':
        ok = mgr.stop_vllm()
        print("✓ vLLM 服务已停止" if ok else "⚠ 未找到运行中的 vLLM 进程")

    elif cmd == 'models':
        lm = mgr.list_local_models()
        print(f"\n本地模型目录: {MODEL_DIR}\n")
        if lm:
            for i, p in enumerate(lm, 1):
                size_mb = sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) // 1_048_576
                print(f"  {i}. {p.name}  ({size_mb:,} MB)")
        else:
            print("  （暂无模型）")
            print()
            print("运行 `sage backend download-guide` 查看下载教程")
        print()

    elif cmd == 'download-guide':
        idx = getattr(args, 'model', None)
        if idx is not None and 1 <= idx <= len(RECOMMENDED_LOCAL_MODELS):
            guide = mgr.model_download_guide(RECOMMENDED_LOCAL_MODELS[idx - 1])
        else:
            # 显示推荐列表
            print("\n推荐本地模型:")
            for i, m in enumerate(RECOMMENDED_LOCAL_MODELS, 1):
                print(f"  {i}. {m['display']:35s}  {m['size']}")
                print(f"     {m['desc']}")
            print()
            raw = input("输入序号查看详细下载教程（Enter 显示通用教程）: ").strip()
            try:
                idx2 = int(raw) - 1
                guide = mgr.model_download_guide(RECOMMENDED_LOCAL_MODELS[idx2])
            except (ValueError, IndexError):
                guide = mgr.model_download_guide()
        print()
        print(guide)

    elif cmd == 'pull':
        mgr.pull_ollama_model(args.model_tag)

    else:
        print(f"未知子命令: {cmd}")
        print("运行 `sage backend --help` 查看帮助")


def setup_skill_parser(subparsers):
    """Setup skill management subcommand."""
    parser = subparsers.add_parser(
        'skill',
        help='管理 seismo_skill 技能文档（查看、搜索、新建、编辑、删除）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
子命令：
  list                    列出所有技能
  search <query>          按关键词搜索技能
  show <name>             显示技能完整文档
  new <name>              新建用户自定义技能（打开编辑器）
  edit <name>             编辑已有用户自定义技能（打开编辑器）
  delete <name>           删除用户自定义技能
  dir                     显示用户技能目录路径

示例:
  python seismic_cli.py skill list
  python seismic_cli.py skill search "带通滤波"
  python seismic_cli.py skill show waveform_processing
  python seismic_cli.py skill new my_tool --title "我的工具" --keywords "tool, 工具"
  python seismic_cli.py skill edit my_tool
  python seismic_cli.py skill delete my_tool
        """
    )
    sub = parser.add_subparsers(dest='skill_cmd', help='技能操作')

    # list
    sub.add_parser('list', help='列出所有技能')

    # search
    sp = sub.add_parser('search', help='按关键词搜索技能')
    sp.add_argument('query', help='搜索关键词（支持中英文）')
    sp.add_argument('-k', '--top-k', type=int, default=3, help='返回结果数（默认 3）')

    # show
    sp = sub.add_parser('show', help='显示技能完整文档')
    sp.add_argument('name', help='技能名称')

    # new
    sp = sub.add_parser('new', help='新建用户自定义技能')
    sp.add_argument('name', help='技能 ID（英文下划线）')
    sp.add_argument('--title', default='', help='中文标题')
    sp.add_argument('--keywords', default='', help='关键词（逗号分隔）')
    sp.add_argument('--desc', default='', help='简短描述')
    sp.add_argument('--editor', default='', help='指定编辑器命令（默认使用 $EDITOR 或 vi）')

    # edit
    sp = sub.add_parser('edit', help='编辑用户自定义技能')
    sp.add_argument('name', help='技能名称')
    sp.add_argument('--editor', default='', help='指定编辑器命令')

    # delete
    sp = sub.add_parser('delete', help='删除用户自定义技能（不可恢复）')
    sp.add_argument('name', help='技能名称')
    sp.add_argument('-y', '--yes', action='store_true', help='跳过确认')

    # dir
    sub.add_parser('dir', help='显示用户技能目录路径')

    return parser


def handle_skill_command(args):
    """Handle skill subcommand."""
    import sys
    from pathlib import Path as _Path

    project_root = str(_Path(__file__).parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from seismo_skill import (
            list_skills, search_skills, load_skill,
            invalidate_cache, skill_loader
        )
        from seismo_skill.skill_loader import (
            save_user_skill, delete_user_skill,
            get_user_skill_dir, get_skill_detail, SKILL_TEMPLATE
        )
    except ImportError as e:
        print(f"✗ 导入 seismo_skill 失败: {e}")
        sys.exit(1)

    cmd = getattr(args, 'skill_cmd', None)
    if cmd is None:
        print("请指定子命令。运行 `python seismic_cli.py skill --help` 查看帮助")
        sys.exit(1)

    # ── list ────────────────────────────────────────────────────────────────
    if cmd == 'list':
        invalidate_cache()
        skills = list_skills()
        if not skills:
            print("（无技能）")
            return
        user_skills    = [s for s in skills if s['source'] == 'user']
        builtin_skills = [s for s in skills if s['source'] == 'builtin']

        col_w = max(len(s['name']) for s in skills) + 2
        header = f"{'名称':{col_w}} {'类别':<14} {'来源':<8} 关键词"
        print(header)
        print('-' * (len(header) + 10))

        if user_skills:
            print("── 自定义 ──")
            for s in user_skills:
                kws = ', '.join(s['keywords'][:3])
                print(f"  ⚡ {s['name']:{col_w-4}} {s['category']:<14} {'[用户]':<8} {kws}")
        if builtin_skills:
            print("── 内置 ──")
            for s in builtin_skills:
                kws = ', '.join(s['keywords'][:3])
                print(f"  📄 {s['name']:{col_w-4}} {s['category']:<14} {'[内置]':<8} {kws}")

        print(f"\n共 {len(skills)} 个技能（{len(user_skills)} 自定义 + {len(builtin_skills)} 内置）")
        print(f"用户技能目录: {get_user_skill_dir()}")

    # ── search ──────────────────────────────────────────────────────────────
    elif cmd == 'search':
        invalidate_cache()
        hits = search_skills(args.query, top_k=args.top_k)
        if not hits:
            print(f"未找到匹配「{args.query}」的技能")
            return
        print(f"搜索「{args.query}」，找到 {len(hits)} 个相关技能：\n")
        for i, s in enumerate(hits, 1):
            src_tag = '[用户]' if s['source'] == 'user' else '[内置]'
            kws = ', '.join(s['keywords'][:4])
            print(f"{i}. {s['name']}  {src_tag}  [{s['category']}]")
            print(f"   关键词: {kws}")
            # Show first 2 lines of body as preview
            preview = s['body'][:200].split('\n')
            for ln in preview[:3]:
                if ln.strip():
                    print(f"   {ln.strip()}")
            print()

    # ── show ────────────────────────────────────────────────────────────────
    elif cmd == 'show':
        text = load_skill(args.name)
        if not text:
            print(f"✗ 未找到技能: {args.name}")
            sys.exit(1)
        detail = get_skill_detail(args.name)
        src_tag = '【自定义】' if detail and detail['source'] == 'user' else '【内置】'
        print(f"{'═'*60}")
        print(f"技能: {args.name}  {src_tag}")
        if detail:
            print(f"文件: {detail['path']}")
        print(f"{'═'*60}\n")
        print(text)

    # ── new ─────────────────────────────────────────────────────────────────
    elif cmd == 'new':
        import os, tempfile, subprocess
        name     = args.name
        title    = args.title or name
        keywords = args.keywords or name
        desc     = args.desc or '自定义技能描述'

        # Check if exists
        if get_skill_detail(name):
            print(f"⚠  技能「{name}」已存在，请用 `skill edit {name}` 修改")
            sys.exit(1)

        tpl = SKILL_TEMPLATE.format(
            name=name, title=title, keywords=keywords, description=desc
        )

        # Write to temp file then open editor
        skill_file = get_user_skill_dir() / f"{name}.md"
        skill_file.write_text(tpl, encoding='utf-8')

        editor = args.editor or os.environ.get('EDITOR', 'vi')
        print(f"✓ 技能模板已创建: {skill_file}")
        print(f"  用编辑器打开: {editor} {skill_file}")
        try:
            subprocess.run([editor, str(skill_file)])
        except FileNotFoundError:
            print(f"  ⚠  编辑器 {editor!r} 未找到，请手动编辑: {skill_file}")
        invalidate_cache()
        print(f"✓ 技能「{name}」已保存到 {skill_file}")

    # ── edit ────────────────────────────────────────────────────────────────
    elif cmd == 'edit':
        import os, subprocess
        detail = get_skill_detail(args.name)
        if detail is None:
            print(f"✗ 未找到技能: {args.name}")
            sys.exit(1)
        if detail['source'] != 'user':
            print(f"✗ 内置技能不可编辑。")
            print(f"  可以复制后另存为自定义技能：")
            print(f"  python seismic_cli.py skill show {args.name} > ~/.seismicx/skills/{args.name}_custom.md")
            sys.exit(1)

        editor = args.editor or os.environ.get('EDITOR', 'vi')
        print(f"打开: {detail['path']}")
        try:
            subprocess.run([editor, detail['path']])
        except FileNotFoundError:
            print(f"⚠  编辑器 {editor!r} 未找到，请手动编辑: {detail['path']}")
        invalidate_cache()
        print("✓ 编辑完成")

    # ── delete ──────────────────────────────────────────────────────────────
    elif cmd == 'delete':
        detail = get_skill_detail(args.name)
        if detail is None:
            print(f"✗ 未找到技能: {args.name}")
            sys.exit(1)
        if detail['source'] != 'user':
            print(f"✗ 内置技能不可删除")
            sys.exit(1)
        if not args.yes:
            confirm = input(f"确认删除自定义技能「{args.name}」？[y/N] ").strip().lower()
            if confirm not in ('y', 'yes'):
                print("已取消")
                return
        ok = delete_user_skill(args.name)
        if ok:
            print(f"✓ 技能「{args.name}」已删除")
        else:
            print(f"✗ 删除失败")
            sys.exit(1)

    # ── dir ─────────────────────────────────────────────────────────────────
    elif cmd == 'dir':
        d = get_user_skill_dir()
        print(f"用户技能目录: {d}")
        import os
        md_files = list(d.glob('*.md'))
        if md_files:
            print(f"当前文件 ({len(md_files)} 个):")
            for f in md_files:
                print(f"  {f.name}")
        else:
            print("（目录为空，尚无自定义技能）")

    else:
        print(f"未知子命令: {cmd}")
        print("运行 `python seismic_cli.py skill --help` 查看帮助")


def setup_agent_parser(subparsers):
    """Setup seismology agent subcommand."""
    parser = subparsers.add_parser(
        'agent',
        help='地震学自主 Agent：阅读文献 → 规划 → 自动编程实现',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 读取 PDF，实现其中的算法
  python seismic_cli.py agent "实现论文中的 HVSR 分析方法" --paper paper.pdf

  # 从 arXiv 获取论文并实现
  python seismic_cli.py agent "复现文中的走时校正方法" --arxiv 2104.12345

  # 不读论文，直接自主规划并实现
  python seismic_cli.py agent "对给定波形计算b值统计并出图" -d /data/

  # 指定输出目录
  python seismic_cli.py agent "实现论文算法" --paper paper.pdf -o results/my_run/
        """
    )
    parser.add_argument('goal', help='自然语言任务目标描述')
    parser.add_argument('--paper', default=None,
                        help='本地 PDF 文件路径')
    parser.add_argument('--arxiv', default=None,
                        help='arXiv 论文 ID（如 2104.12345）或完整 URL')
    parser.add_argument('--doi', default=None,
                        help='DOI（如 10.1029/2022JB024987）')
    parser.add_argument('--text', default=None,
                        help='直接提供论文摘要或方法描述文本')
    parser.add_argument('-d', '--data', default=None,
                        help='数据目录或文件（Agent 执行时使用）')
    parser.add_argument('-o', '--output', default=None,
                        help='输出目录（默认自动生成时间戳目录）')
    parser.add_argument('--max-steps', type=int, default=8,
                        help='最大执行步骤数（默认 8）')
    return parser


def run_agent(args):
    """Run the seismology agent."""
    import sys
    from pathlib import Path as _Path
    from datetime import datetime as _dt

    project_root = str(_Path(__file__).parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from seismo_agent.agent_loop import SeismoAgent
    except ImportError as e:
        print(f"✗ 导入 seismo_agent 模块失败: {e}")
        sys.exit(1)

    # Output directory
    if args.output:
        output_dir = args.output
    else:
        results_dir = _Path(project_root) / 'results'
        results_dir.mkdir(exist_ok=True)
        ts = _dt.now().strftime('%Y%m%d_%H%M%S')
        output_dir = str(results_dir / f'agent_{ts}')

    agent = SeismoAgent(project_root=project_root)

    if not agent.is_llm_available():
        print("✗ LLM 服务不可用。")
        print("  - Ollama: 运行 `ollama serve` 后重试")
        print("  - API: 运行 `python seismic_cli.py llm setup` 配置")
        sys.exit(1)

    # Determine paper source
    paper_source = None
    if args.paper:
        paper_source = args.paper
    elif args.arxiv:
        paper_source = args.arxiv
    elif args.doi:
        paper_source = args.doi
    elif args.text:
        paper_source = args.text

    # Data hint
    goal = args.goal
    if args.data:
        goal += f"\n数据路径: {args.data}"

    print(f"任务: {goal}")
    if paper_source:
        print(f"文献: {paper_source}")
    print(f"输出: {output_dir}")
    print("-" * 60)

    result = agent.run(
        goal=goal,
        paper_source=paper_source,
        output_dir=output_dir,
        progress_cb=print,
        max_steps=args.max_steps,
    )

    print("-" * 60)
    if result["success"]:
        print(f"✓ Agent 完成，共 {result['steps_completed']}/{result['steps_total']} 步成功")
    else:
        print(f"⚠  部分步骤失败，{result['steps_completed']}/{result['steps_total']} 步成功")
    print(f"输出目录: {result['output_dir']}")


def main():
    # Check for first run and LLM configuration
    config = get_config_manager()
    if config.is_first_run():
        print("=" * 80)
        print("Welcome to SeismicX!")
        print("=" * 80)
        print()
        print("This appears to be your first time using SeismicX.")
        print("Let's configure your LLM (Large Language Model) settings.")
        print()
        print("You can:")
        print("  - Use Ollama for local models (recommended for privacy)")
        print("  - Use online APIs like OpenAI GPT-4 or Anthropic Claude")
        print()

        setup_choice = input("Would you like to set up LLM configuration now? (y/n) [y]: ").strip().lower()
        if setup_choice != 'n':
            config.interactive_setup()
        else:
            config.mark_first_run_complete()
            print()
            print("You can configure LLM settings later with:")
            print("  python seismic_cli.py llm setup")

        print()
        print("=" * 80)
        print()

    parser = argparse.ArgumentParser(
        description='SeismicX CLI - Unified interface for seismic analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Configure LLM settings
  python seismic_cli.py llm setup
  python seismic_cli.py llm show
  python seismic_cli.py llm list-models

  # Phase picking
  python seismic_cli.py pick -i data/waveforms -o results/picks -m pnsn/pickers/pnsn.v3.jit -d cpu

  # Phase association with FastLink
  python seismic_cli.py associate -i results/picks.txt -o results/events.txt -s data/stations.txt -m fastlink

  # Phase association with REAL
  python seismic_cli.py associate -i results/picks.txt -o results/events.txt -s data/stations.txt -m real

  # Polarity analysis
  python seismic_cli.py polarity -i results/picks.txt -w data/waveforms -o results/polarity.txt

  # LLM 代码生成与执行
  python seismic_cli.py run "对 /data/wave.mseed 做 1-10Hz 带通滤波并画图"
  python seismic_cli.py run "计算震源参数" -d /data/waveforms/

  # 外部工具管理
  python seismic_cli.py tool list
  python seismic_cli.py tool info hypodd
  python seismic_cli.py tool parse /path/to/readme.txt
  python seismic_cli.py tool gen hypodd -d /data/ -o /work/hypodd_run/

  # Seismological statistics (b-value, Mc, G-R plot)
  python seismic_cli.py stats -i results/
  python seismic_cli.py stats -i catalog.csv --mc 2.0 --method mle --plot
  python seismic_cli.py stats -i catalog.json -o results/stats_run1 --plot

For more information, see the skill documentation in .lingma/skills/
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Setup subcommands
    setup_llm_parser(subparsers)
    setup_backend_parser(subparsers)
    setup_pick_parser(subparsers)
    setup_associate_parser(subparsers)
    setup_polarity_parser(subparsers)
    setup_stats_parser(subparsers)
    setup_run_parser(subparsers)
    setup_tool_parser(subparsers)
    setup_agent_parser(subparsers)
    setup_skill_parser(subparsers)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Execute appropriate command
    if args.command == 'llm':
        handle_llm_command(args)
    elif args.command == 'pick':
        run_picking(args)
    elif args.command == 'associate':
        run_association(args)
    elif args.command == 'polarity':
        run_polarity(args)
    elif args.command == 'stats':
        run_stats(args)
    elif args.command == 'run':
        run_code(args)
    elif args.command == 'tool':
        handle_tool_command(args)
    elif args.command == 'backend':
        handle_backend_command(args)
    elif args.command == 'agent':
        run_agent(args)
    elif args.command == 'skill':
        handle_skill_command(args)


if __name__ == '__main__':
    main()
