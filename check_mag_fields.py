import ujson as json
import collections
import statistics

import tqdm


if __name__ == '__main__':

    query_paper_count_by_mag_field = collections.defaultdict(int)
    pos_paper_count_by_mag_field = collections.defaultdict(int)

    num_query_paper_ids_found_mag = 0
    num_pos_paper_ids_found_mag = 0

    num_triples_pos_cross_domain_pure = 0
    num_triples_neg_cross_domain_pure = 0

    num_triples_pos_cross_domain = 0
    num_triples_neg_cross_domain = 0

    unique_paper_ids = set()

    extra_metadata = json.load(open('mag_fields_by_all_paper_ids.json', 'r'))

    with open("cite/test.qrel", 'r') as f_in:
        lines = f_in.readlines()

        for l in tqdm.tqdm(lines):

            l_parsed = l.split()

            query_paper_id = l_parsed[0]
            pos_paper_id = l_parsed[2]

            unique_paper_ids.add(query_paper_id)
            unique_paper_ids.add(pos_paper_id)

            if query_paper_id in extra_metadata.keys() and len(extra_metadata[query_paper_id]) > 0:
                num_query_paper_ids_found_mag += 1

                for f in extra_metadata[query_paper_id]:
                    query_paper_count_by_mag_field[f] += 1
            else:
                query_paper_count_by_mag_field['**Unknown**'] += 1

            if pos_paper_id in extra_metadata.keys() and len(extra_metadata[pos_paper_id]) > 0:
                num_pos_paper_ids_found_mag += 1

                if query_paper_id in extra_metadata.keys() and len(extra_metadata[query_paper_id]) > 0:
                    if len(set(extra_metadata[query_paper_id]).intersection(set(extra_metadata[pos_paper_id]))) == 0:
                        num_triples_pos_cross_domain_pure += 1

                    if len(set(extra_metadata[pos_paper_id]) - set(extra_metadata[query_paper_id])) > 0:
                        num_triples_pos_cross_domain += 1

                for f in extra_metadata[pos_paper_id]:
                    pos_paper_count_by_mag_field[f] += 1
            else:
                pos_paper_count_by_mag_field['**Unknown**'] += 1
