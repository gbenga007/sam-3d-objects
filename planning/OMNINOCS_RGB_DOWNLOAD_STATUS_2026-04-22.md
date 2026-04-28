# OmniNOCS RGB Download Status - 2026-04-22

## Context

OmniNOCS provides NOCS maps, instance maps, and object metadata, but it does not
bundle source RGB images for all constituent datasets. The OmniNOCS setup guide
points users to the original dataset image sources. For the datasets we want next:

- Objectron RGB can be obtained from the Omni3D preprocessed image zip.
- ARKitScenes RGB can be obtained from the Omni3D preprocessed image zip.
- Hypersim RGB should use the `.tonemap.jpg` preview image download path from
  the Hypersim tools.

These RGB images are only used as source images. Training still uses OmniNOCS
metadata and masks.

## Local Paths

Current OmniNOCS root:

```text
/mnt/dest/OmniNOCS
```

Download scripts copied to:

```text
/mnt/dest/OmniNOCS/download_scripts/download_objectron_images.sh
/mnt/dest/OmniNOCS/download_scripts/download_arkitscenes_images.sh
```

RGB download working directory:

```text
/mnt/dest/OmniNOCS/omni3d_rgb
```

## Commands Started

The following resumable downloads were started from
`/mnt/dest/OmniNOCS/omni3d_rgb`:

```bash
wget -c https://dl.fbaipublicfiles.com/omni3d_data/objectron_images.zip
wget -c https://dl.fbaipublicfiles.com/omni3d_data/ARKitScenes_images.zip
```

The original Omni3D scripts are simple wrappers equivalent to:

```bash
wget https://dl.fbaipublicfiles.com/omni3d_data/objectron_images.zip
unzip objectron_images.zip

wget https://dl.fbaipublicfiles.com/omni3d_data/ARKitScenes_images.zip
unzip ARKitScenes_images.zip
```

## Status At Interruption

The user interrupted the long-running monitor command, but the `wget` processes
were still active when checked immediately afterward:

```text
PID 95942 wget -c https://dl.fbaipublicfiles.com/omni3d_data/objectron_images.zip
PID 95943 wget -c https://dl.fbaipublicfiles.com/omni3d_data/ARKitScenes_images.zip
```

Partial files present at that check:

```text
/mnt/dest/OmniNOCS/omni3d_rgb/objectron_images.zip       ~2.3G of ~23G
/mnt/dest/OmniNOCS/omni3d_rgb/ARKitScenes_images.zip     ~715M of ~27G
```

Available storage before starting:

```text
/mnt/dest: 698G available
```

## Status Update - 2026-04-23 01:51

Objectron and ARKitScenes zips completed and were extracted with:

```bash
cd /mnt/dest/OmniNOCS/omni3d_rgb
unzip -n objectron_images.zip
unzip -n ARKitScenes_images.zip
```

Extracted layouts observed:

```text
/mnt/dest/OmniNOCS/omni3d_rgb/datasets/objectron/train/book_batch_11_29_0000200.jpg
/mnt/dest/OmniNOCS/omni3d_rgb/datasets/ARKitScenes/Training/42444477/1532.580_00004419.jpg
```

Hypersim is still downloading via:

```text
python /mnt/dest/OmniNOCS/ml-hypersim/contrib/99991/download.py -c .tonemap.jpg -d /mnt/dest/OmniNOCS/omni3d_rgb/hypersim --silent
```

Observed Hypersim layout:

```text
/mnt/dest/OmniNOCS/omni3d_rgb/hypersim/ai_001_001/images/scene_cam_00_final_preview/frame.0000.tonemap.jpg
```

`sam3d_objects/data/dataset/metric/omninocs.py` was updated to resolve these
layouts:

- Objectron flattened Omni3D names from OmniNOCS metadata paths.
- ARKitScenes `datasets/ARKitScenes/...` names, preserving timestamped basenames.
- Hypersim `scene_cam_XX_final_preview/frame.NNNN.tonemap.jpg` names.
- RGB images are resized to the instance-mask resolution before RGBA composition
  when source RGB and OmniNOCS masks have different spatial sizes.

Bounded source smoke tests passed:

```text
Objectron:   10 records, example book/batch-13/3/frame000000, image shape (480, 360, 4)
ARKitScenes: 10 records, example ARKitScenes/Training/42444477/1532.580_00004419, image shape (256, 192, 4)
Hypersim:    5 records, example ai_022_004/cam_01/frame_0006, image shape (768, 1024, 4)
```

Combined four-source smoke test also passed:

```text
len 20
sources [('arkitscenes', 5), ('hypersim', 5), ('nocs_real275', 5), ('objectron', 5)]
first nocs_real275 nocs_real275/test/scene_1/0066 (480, 640, 4)
```

Targeted Hypersim follow-up:

- Full `.tonemap.jpg` crawl was not a good fit for resume/recovery.
- A targeted helper was added:

```text
scripts/download_targeted_hypersim_rgb.py
```

- Current OmniNOCS Hypersim reference counts:

```text
unique_frames   50319
existing        33260
missing         17059
missing_scenes  88
missing_cameras 181
```

- The targeted strategy groups missing RGB by `scene/camera` and downloads
  entire missing preview camera trajectories with the official Hypersim helper.

Targeted download command now in use:

```bash
python scripts/download_targeted_hypersim_rgb.py
```

Initial confirmation after launch:

```text
before targeted fetch: 51512 files
after initial targeted fetch check: 51580 files
new files in last 2 minutes: 72
```

This confirms the targeted camera-level fetch is making forward progress, unlike
the previous full `.tonemap.jpg` crawl.

Disk after extracting Objectron and ARKitScenes while Hypersim continued:

```text
/mnt/dest: 594G available
```

## Resume / Continue

If the downloads are no longer running, resume them with:

```bash
cd /mnt/dest/OmniNOCS/omni3d_rgb
wget -c https://dl.fbaipublicfiles.com/omni3d_data/objectron_images.zip
wget -c https://dl.fbaipublicfiles.com/omni3d_data/ARKitScenes_images.zip
```

After each zip completes:

```bash
cd /mnt/dest/OmniNOCS/omni3d_rgb
unzip objectron_images.zip
unzip ARKitScenes_images.zip
```

Then inspect the extracted folder layout and update
`sam3d_objects/data/dataset/metric/omninocs.py` if the RGB path resolver needs to
match the Omni3D directory names.

## Hypersim Status

Hypersim was started after Objectron and ARKitScenes using the official
`ml-hypersim` helper from Thomas Germer's `contrib/99991` path.

Tool repo:

```text
/mnt/dest/OmniNOCS/ml-hypersim
```

Destination:

```text
/mnt/dest/OmniNOCS/omni3d_rgb/hypersim
```

Command started:

```bash
git clone https://github.com/apple/ml-hypersim /mnt/dest/OmniNOCS/ml-hypersim
cd /mnt/dest/OmniNOCS/ml-hypersim
python contrib/99991/download.py -c .tonemap.jpg -d /mnt/dest/OmniNOCS/omni3d_rgb/hypersim --silent
```

The Hypersim process was active when checked:

```text
python /mnt/dest/OmniNOCS/ml-hypersim/contrib/99991/download.py -c .tonemap.jpg -d /mnt/dest/OmniNOCS/omni3d_rgb/hypersim --silent
```

Initial destination layout had begun with:

```text
/mnt/dest/OmniNOCS/omni3d_rgb/hypersim/ai_001_001
```

After download, verify that paths can satisfy OmniNOCS metadata examples such as:

```text
ai_050_004/cam_04/frame_0047
ai_022_004/cam_01/frame_0006
ai_027_001/cam_01/frame_0006
```

## Loader Smoke Tests To Run Next

Bounded source checks:

```bash
env LIDRA_SKIP_INIT=1 python -c "from sam3d_objects.data.dataset.metric import OmniNOCSObjectDataset; d=OmniNOCSObjectDataset(sources=['objectron'], rgb_roots={'objectron':'/mnt/dest/OmniNOCS/omni3d_rgb'}, max_records_per_source=10); print(len(d), d[0]['source'], d[0]['image_name'])"
```

```bash
env LIDRA_SKIP_INIT=1 python -c "from sam3d_objects.data.dataset.metric import OmniNOCSObjectDataset; d=OmniNOCSObjectDataset(sources=['arkitscenes'], rgb_roots={'arkitscenes':'/mnt/dest/OmniNOCS/omni3d_rgb'}, max_records_per_source=10); print(len(d), d[0]['source'], d[0]['image_name'])"
```

If these fail due to file layout mismatch, inspect a few extracted image paths
with `find /mnt/dest/OmniNOCS/omni3d_rgb -maxdepth 5 -type f | head` and add
candidate path patterns to `_rgb_path` in the mixed OmniNOCS loader.

The Objectron, ARKitScenes, and Hypersim checks above now pass. Next checks:

```bash
env LIDRA_SKIP_INIT=1 python -c "from sam3d_objects.data.dataset.metric import OmniNOCSObjectDataset; d=OmniNOCSObjectDataset(sources=['nocs_real275','objectron','arkitscenes','hypersim'], rgb_roots={'nocs_real275':'/mnt/dest/OmniNOCS/real_test','objectron':'/mnt/dest/OmniNOCS/omni3d_rgb','arkitscenes':'/mnt/dest/OmniNOCS/omni3d_rgb','hypersim':'/mnt/dest/OmniNOCS/omni3d_rgb/hypersim'}, max_records_per_source=5); print(len(d)); print(sorted({r['source'] for r in d.records}))"
```
