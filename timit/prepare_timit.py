from lhotse.recipes.timit import download_timit, prepare_timit
from lhotse import CutSet
from collections import defaultdict
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet


# TIMIT_DEFAULT = "/home/acp20rm/data/timit/"
TIMIT_DEFAULT = "/store/store1/data/"

def get_manifests_from_file(strManifestPath = TIMIT_DEFAULT):
    
    manifests = defaultdict(dict)
    for split in ["TRAIN", "DEV", "TEST"]:
        recording_set = RecordingSet.from_file(f'{strManifestPath}/timit_dataset/timit_recordings_{split}.jsonl.gz')
        supervision_set = SupervisionSet.from_file(f'{strManifestPath}/timit_dataset/timit_supervisions_{split}.jsonl.gz')

        manifests[split] = {
                        "recordings": recording_set,
                        "supervisions": supervision_set,
                    }
    
    return manifests

if __name__ == '__main__':

    print("Preparing TIMIT")

    # download_timit(
    #     target_dir="/home/acp20rm/data/"
    # )
    manifests = prepare_timit(
        corpus_dir=f'{TIMIT_DEFAULT}/TIMIT',
        output_dir=f'{TIMIT_DEFAULT}/timit_dataset'
    )

    # manifests = get_manifests_from_file()

    splits = ["TRAIN", "DEV", "TEST"]
    for split in splits:
        print(manifests[split])


    cuts = defaultdict(dict)

    for split in splits:
        cuts[split] = CutSet.from_manifests(
            recordings = manifests[split]['recordings'],
            supervisions = manifests[split]['supervisions']
            )
        cuts[split].to_file(f'{TIMIT_DEFAULT}/timit_dataset/timit_cuts_{split}.jsonl.gz')

