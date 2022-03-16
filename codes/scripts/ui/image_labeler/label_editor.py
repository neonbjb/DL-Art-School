import orjson

from data.images.image_label_parser import VsNetImageLabeler


# Translates from the label JSON output of the VS.NET UI to something more compact and usable.
def convert_from_vsnet_labels():
    labeler = VsNetImageLabeler(['F:\\4k6k\datasets\\ns_images\\512_unsupervised\\categories.json',
                                 'F:\\4k6k\datasets\\ns_images\\512_unsupervised\\categories_new.json',
                                 'F:\\4k6k\datasets\\ns_images\\512_unsupervised\\categories_new_new.json'])
    # Proposed format:
    # 'config': { 'dim' }
    # 'labels': [{ 'label', 'key'}] <- ordered by label index.
    # 'images': {'file': [{ 'lid', 'top', 'left' }}
    # 'labelMap' {<mapping of string labels to ids>}
    out_dict = {
        'config': {
            'dim': next(iter(labeler.labeled_images.values()))[0]['patch_width']
        },
        'labels': [{'label': cat['label'], 'key': cat['keyBinding']} for cat in labeler.categories.values()],
    }
    out_dict['labelMap'] = {}
    for i, lbl in enumerate(out_dict['labels']):
        out_dict['labelMap'][lbl['label']] = i
    out_dict['images'] = {}
    for fname, ilbls in labeler.labeled_images.items():
        out_dict['images'][fname] = [{'lid': out_dict['labelMap'][il['label']], 'top': il['patch_top'], 'left': il['patch_left']} for il in ilbls]
    with open("label_editor.json", 'wb') as fout:
        fout.write(orjson.dumps(out_dict))


if __name__ == '__main__':
    convert_from_vsnet_labels()