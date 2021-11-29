from types import resolve_bases
from inference import get_model
import os
import json


def run(jObj):
    output = dict()
    model = get_model()
    
    cancer = list()
    cancer_not = list()
    a = 0
    out_dict = {}
    for note in jObj['notes'][:10]:
        result_dict = model.predict(note['textHtml'])
        if len(result_dict[1]) !=0:
            date = note['encounterDate']
            out_dict[a] = {'encounterDate':date, 'results': result_dict}
            a += 1

    return out_dict

if __name__ == "__main__":
    cur_path = os.path.abspath(os.path.dirname(__file__))
    eval_path = os.path.join(os.path.join(cur_path, 'data'), 'eval_data')
    eval_files = [file for file in os.listdir(eval_path) if file.endswith('.json') and not file.endswith('_output.json')]
    example_note_path = [os.path.join(eval_path, i) for i in eval_files]
    print('kk',example_note_path)
    for i in range(len(example_note_path)):

        with open(example_note_path[i], 'r') as fp:
            jObj = json.load(fp)
        output = run(jObj)
    
        # debug
        with open(os.path.join(eval_path, eval_files[i].replace('.json', '')+'_output.json'), "w") as f:
            json.dump(output, f, indent=4)

    print(output)




