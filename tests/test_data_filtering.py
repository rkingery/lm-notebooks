import tempfile
from pathlib import Path

from .common import FIXTURES_PATH

def test_extract_text_from_html_bytes(run_extract_text_from_html_bytes):
    with open(FIXTURES_PATH / 'moby.html', 'rb') as f:
        moby_bytes = f.read()
    with open(FIXTURES_PATH / 'moby_extracted.txt') as f:
        moby_expected_text = f.read()
    assert moby_expected_text == run_extract_text_from_html_bytes(moby_bytes)
    print('PASSED: test_extract_text_from_html_bytes')

def test_identify_language_english(run_identify_language):
    with open(FIXTURES_PATH / 'moby_extracted.txt') as f:
        moby_expected_text = f.read()
    predicted_language, score = run_identify_language(moby_expected_text)
    assert predicted_language == 'en'
    assert isinstance(score, float)
    assert score > 0
    print('PASSED: test_identify_language_english')

def test_identify_language_chinese_simplified(run_identify_language):
    predicted_language, score = run_identify_language('欢迎来到我们的网站')
    assert predicted_language == 'zh'
    assert isinstance(score, float)
    assert score > 0
    print('PASSED: test_identify_language_chinese_simplified')

def test_mask_emails_single(run_mask_emails):
    test_string = 'Feel free to contact me at test@gmail.com if you have any questions.'
    expected = 'Feel free to contact me at |||EMAIL_ADDRESS||| if you have any questions.'
    masked_text, num_masked = run_mask_emails(test_string)
    assert masked_text == expected
    assert num_masked == 1
    print('PASSED: test_mask_emails_single')

def test_mask_emails_multiple(run_mask_emails):
    test_string = 'The instructors are pl@fakedomain.ai and spl@fakedomain.ai'
    expected = 'The instructors are |||EMAIL_ADDRESS||| and |||EMAIL_ADDRESS|||'
    masked_text, num_masked = run_mask_emails(test_string)
    assert masked_text == expected
    assert num_masked == 2
    print('PASSED: test_mask_emails_multiple')

def test_mask_emails_existing_string(run_mask_emails):
    test_string = 'Some datasets use the string |||EMAIL_ADDRESS||| to represent masked PII. The instructors are pl@fakedomain.ai and spl@fakedomain.ai'
    expected = 'Some datasets use the string |||EMAIL_ADDRESS||| to represent masked PII. The instructors are |||EMAIL_ADDRESS||| and |||EMAIL_ADDRESS|||'
    masked_text, num_masked = run_mask_emails(test_string)
    assert masked_text == expected
    assert num_masked == 2
    print('PASSED: test_mask_emails_existing_string')

def test_mask_phones_single(run_mask_phone_numbers):
    numbers = ['2831823829', '(283)-182-3829', '(283) 182 3829', '283-182-3829']
    for number in numbers:
        test_string = f'Feel free to contact me at {number} if you have any questions.'
        expected = 'Feel free to contact me at |||PHONE_NUMBER||| if you have any questions.'
        masked_text, num_masked = run_mask_phone_numbers(test_string)
        assert masked_text == expected, f'Failed for number format: {number}'
        assert num_masked == 1
    print('PASSED: test_mask_phones_single')

def test_mask_ips(run_mask_ips):
    test_string = 'You can access the server at 192.0.2.146.'
    expected = 'You can access the server at |||IP_ADDRESS|||.'
    masked_text, num_masked = run_mask_ips(test_string)
    assert masked_text == expected
    assert num_masked == 1
    print('PASSED: test_mask_ips')

def test_classify_nsfw(run_classify_nsfw):
    prediction, score = run_classify_nsfw(
        'SUCK MY C*CK WIKIPEDIA EDITORS...F*CKING *SSH*LE DORKS. JUST TRYING TO MAKE THE SITE BETTER YOU UPTIGHT C*NTS'
    )
    assert prediction == 'nsfw'
    assert isinstance(score, float)
    assert score > 0
    prediction, score = run_classify_nsfw('Umm, theres no actual article for prostitution ring.  - Crunch Captain.')
    assert prediction == 'non-nsfw'
    assert isinstance(score, float)
    assert score > 0
    print('PASSED: test_classify_nsfw')

def test_classify_toxic_speech(run_classify_toxic_speech):
    prediction, score = run_classify_toxic_speech(
        "Why did that idiot revert the reversion I made? Can that moron not have the decent common manners to post on the talk page? "
        "What a rude fuck. Arrogant twat who doesn't know what he's talking about. None of you fuckers have any manners."
    )
    assert prediction == 'toxic'
    assert isinstance(score, float)
    assert score > 0
    prediction, score = run_classify_toxic_speech('Why the fc*k should I get a warning for doing nothing?')
    assert prediction == 'non-toxic'
    assert isinstance(score, float)
    assert score > 0
    print('PASSED: test_classify_toxic_speech')

def test_classify_quality(run_classify_quality):
    with open(FIXTURES_PATH / 'low_quality_cc.txt') as f:
        low_quality_cc = f.read()
    prediction, score = run_classify_quality(low_quality_cc)
    assert prediction == 'cc'
    assert isinstance(score, float)
    assert score > 0
    with open(FIXTURES_PATH / 'high_quality_wiki_reference.txt') as f:
        high_quality_wiki = f.read()
    prediction, score = run_classify_quality(high_quality_wiki)
    assert prediction == 'wiki'
    assert isinstance(score, float)
    assert score > 0
    print('PASSED: test_classify_quality')

def test_gopher_valid_input(run_gopher_quality_filter):
    text = 'This should definitely be a valid input text and of high quality according to Gopher rules. ' * 100
    assert run_gopher_quality_filter(text)
    print('PASSED: test_gopher_valid_input')

def test_gopher_less_than_50_non_symbol_words(run_gopher_quality_filter):
    text = 'The string you are reading is a short snippet of text.'
    assert not run_gopher_quality_filter(text)
    text = 'The string you are reading is a long snippet of text.' * 100
    assert run_gopher_quality_filter(text)
    print('PASSED: test_gopher_less_than_50_non_symbol_words')

def test_gopher_more_than_100000_non_symbol_words(run_gopher_quality_filter):
    text = 'The string you are reading is too long of a text. ' * 50000
    assert not run_gopher_quality_filter(text)
    text = 'The string you are reading is an okay example of text. ' * 5000
    assert run_gopher_quality_filter(text)
    print('PASSED: test_gopher_more_than_100000_non_symbol_words')

def test_gopher_average_word_length_less_than_3(run_gopher_quality_filter):
    text = 'the be ' * 100
    assert not run_gopher_quality_filter(text)
    text = 'the with ' * 100
    assert run_gopher_quality_filter(text)
    print('PASSED: test_gopher_average_word_length_less_than_3')

def test_gopher_average_word_length_greater_than_10(run_gopher_quality_filter):
    text = 'the and extraordinarily extraordinarily extraordinarily longesest ' * 100
    assert not run_gopher_quality_filter(text)
    text = 'the and this is fine ' * 100
    assert run_gopher_quality_filter(text)
    print('PASSED: test_gopher_average_word_length_greater_than_10')

def test_gopher_more_than_30_percent_lines_ending_with_ellipsis(run_gopher_quality_filter):
    lines = ['The line here is an example of line ending with an ellipsis...' for _ in range(70)]
    lines += ['This is a normal line.' for _ in range(30)]
    text = '\n'.join(lines)
    assert not run_gopher_quality_filter(text)
    lines = ['The line here is an example of ending with ellipsis...' for _ in range(30)]
    lines += ['This is a normal line.' for _ in range(230)]
    text = '\n'.join(lines)
    assert run_gopher_quality_filter(text)
    print('PASSED: test_gopher_more_than_30_percent_lines_ending_with_ellipsis')

def test_gopher_less_than_80_percent_words_with_alphabetic_character(run_gopher_quality_filter):
    words = ['123' for _ in range(8)]
    words += ['word' for _ in range(2)]
    text = 'the and ' + ' '.join(words)
    assert not run_gopher_quality_filter(text)
    print('PASSED: test_gopher_less_than_80_percent_words_with_alphabetic_character')

def test_exact_line_deduplication(run_exact_line_deduplication):
    documents_with_line_duplicates_paths = list((FIXTURES_PATH / 'documents_with_line_duplicates').glob('doc*.txt'))
    documents_without_line_duplicates_paths = list((FIXTURES_PATH / 'documents_line_deduplicated').glob('doc*.txt'))
    deduplicated_documents = []
    for path in documents_without_line_duplicates_paths:
        with open(path) as f:
            deduplicated_documents.append(f.read())
    tmp_path = Path(tempfile.mkdtemp())
    run_exact_line_deduplication(input_files=documents_with_line_duplicates_paths, output_directory=tmp_path)
    output_filepaths = list(tmp_path.glob('*'))
    assert len(output_filepaths) == 5
    for filepath in output_filepaths:
        with open(filepath) as f:
            output_file_contents = f.read()
            try:
                deduplicated_documents.remove(output_file_contents)
            except ValueError:
                raise ValueError(
                    f'Failed to find output file {filepath} contents {output_file_contents.__repr__()} in '
                    f'expected deduplicated documents {deduplicated_documents}.'
                )
    assert len(deduplicated_documents) == 0
    print('PASSED: test_exact_line_deduplication')

def test_minhash_deduplication_exact_duplicates(run_minhash_deduplication):
    documents_with_line_duplicates_paths = list((FIXTURES_PATH / 'documents_with_line_duplicates').glob('doc*.txt'))
    deduplicated_documents = []
    for path in documents_with_line_duplicates_paths:
        if path.name == 'doc2.txt':
            continue
        with open(path) as f:
            deduplicated_documents.append(f.read())
    tmp_path = Path(tempfile.mkdtemp())
    run_minhash_deduplication(input_files=documents_with_line_duplicates_paths, output_directory=tmp_path, num_hashes=100, num_bands=10,
                              ngrams=5, jaccard_threshold=0.8)
    output_filepaths = list(tmp_path.glob('*'))
    assert len(output_filepaths) == 4
    for filepath in output_filepaths:
        with open(filepath) as f:
            output_file_contents = f.read()
            try:
                deduplicated_documents.remove(output_file_contents)
            except ValueError:
                raise ValueError(
                    f'Failed to find output file {filepath} contents {output_file_contents.__repr__()} in '
                    f'expected deduplicated documents {deduplicated_documents}.'
                )
    assert len(deduplicated_documents) == 0
    print('PASSED: test_minhash_deduplication_exact_duplicates')

def test_minhash_deduplication_fuzzy_duplicates(run_minhash_deduplication):
    documents_with_fuzzy_duplicates_paths = list((FIXTURES_PATH / 'documents_with_fuzzy_duplicates').glob('*.txt'))
    deduplicated_documents = []
    kept_duplicated_documents = []
    for path in documents_with_fuzzy_duplicates_paths:
        with open(path) as f:
            if path.name == 'rails_mit_license.txt' or path.name == 'react_mit_license.txt':
                kept_duplicated_documents.append(f.read())
            else:
                deduplicated_documents.append(f.read())
    tmp_path = Path(tempfile.mkdtemp())
    run_minhash_deduplication(input_files=documents_with_fuzzy_duplicates_paths, output_directory=tmp_path, num_hashes=500, num_bands=50,
                              ngrams=5, jaccard_threshold=0.8)
    output_filepaths = list(tmp_path.glob('*'))
    assert len(output_filepaths) == 2
    for filepath in output_filepaths:
        with open(filepath) as f:
            output_file_contents = f.read()
            if output_file_contents in deduplicated_documents:
                deduplicated_documents.remove(output_file_contents)
            elif output_file_contents in kept_duplicated_documents:
                kept_duplicated_documents.remove(output_file_contents)
            else:
                raise ValueError(
                    f'Failed to find output file {filepath} contents {output_file_contents.__repr__()} in '
                    f'expected deduplicated documents {deduplicated_documents} or '
                    f'kept duplicated documents {kept_duplicated_documents}.'
                )
    assert len(deduplicated_documents) == 0
    assert len(kept_duplicated_documents) == 1
    print('PASSED: test_minhash_deduplication_fuzzy_duplicates')
