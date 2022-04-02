import json
import collections
import statistics

import tqdm


if __name__ == '__main__':

    query_paper_count_by_mag_field = collections.defaultdict(int)
    pos_paper_count_by_mag_field = collections.defaultdict(int)

    num_query_paper_ids_found_mag = 0
    num_pos_paper_ids_found_mag = 0

    unique_paper_ids = set()

    extra_metadata = json.load(open('/mnt/nfs/scratch1/bseoh/scincl_dataset/scidocs_extra_metadata.json', 'r'))

    with open("data/cite/test.qrel", 'r') as f_in:
        lines = f_in.readlines()

        for l in tqdm.tqdm(lines):

            l_parsed = l.split()

            query_paper_id = l_parsed[0]
            pos_paper_id = l_parsed[2]

            unique_paper_ids.add(query_paper_id)
            unique_paper_ids.add(pos_paper_id)

            if query_paper_id in extra_metadata.keys() and extra_metadata[query_paper_id]['mag_field_of_study'] is not None:
                num_query_paper_ids_found_mag += 1

                for f in extra_metadata[query_paper_id]['mag_field_of_study']:
                    query_paper_count_by_mag_field[f] += 1
            else:
                query_paper_count_by_mag_field['**Unknown**'] += 1

            if pos_paper_id in extra_metadata.keys() and extra_metadata[pos_paper_id]['mag_field_of_study'] is not None:
                num_pos_paper_ids_found_mag += 1

                for f in extra_metadata[pos_paper_id]['mag_field_of_study']:
                    pos_paper_count_by_mag_field[f] += 1
            else:
                pos_paper_count_by_mag_field['**Unknown**'] += 1
