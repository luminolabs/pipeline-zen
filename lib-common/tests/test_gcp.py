from unittest.mock import patch

import pytest
from google.api_core.exceptions import TooManyRequests

from common.gcp import (
    insert_to_biqquery,
    publish_to_pubsub,
    get_results_bucket,
    upload_directory,
    upload_file,
    upload_jobs_meta,
    make_object_public,
    download_object,
    get_vm_name_from_metadata,
    get_zone_from_metadata,
    get_mig_name_from_vm_name,
    get_region_from_zone,
    get_multi_region_from_zone,
    get_region_from_vm_name
)


@pytest.fixture
def mock_config():
    with patch('common.gcp.config') as mock_cfg:
        mock_cfg.send_to_bq = True
        mock_cfg.send_to_pubsub = True
        mock_cfg.send_to_gcs = True
        mock_cfg.gcp_project = 'test-project'
        mock_cfg.bq_dataset = 'test_dataset'
        mock_cfg.jobs_meta_topic = 'test-topic'
        mock_cfg.is_gcp = True
        mock_cfg.local_env_name = 'local'
        mock_cfg.env_name = 'test'
        mock_cfg.results_bucket_suffix = None
        mock_cfg.job_meta_file = 'job-meta.json'
        mock_cfg.zone = None
        mock_cfg.vm_name = None
        yield mock_cfg


@pytest.fixture
def mock_storage_client():
    with patch('common.gcp.storage_client') as mock_client:
        yield mock_client


@pytest.fixture
def mock_bigquery_client():
    with patch('common.gcp.bigquery_client') as mock_client:
        yield mock_client


@pytest.fixture
def mock_pubsub_client():
    with patch('common.gcp.pubsub_publisher_client') as mock_client:
        yield mock_client


@pytest.fixture
def mock_requests():
    with patch('common.gcp.requests') as mock_req:
        yield mock_req


# Test BigQuery Operations
def test_insert_to_bigquery(mock_bigquery_client, mock_config):
    """Test successful BigQuery insertion"""
    table = 'test-project.test_dataset.test_table'
    row = {'field1': 'value1', 'field2': 'value2'}

    mock_bigquery_client.insert_rows_json.return_value = []

    insert_to_biqquery(table, row)

    mock_bigquery_client.insert_rows_json.assert_called_once_with(table, [row])


def test_insert_to_bigquery_error(mock_bigquery_client, mock_config):
    """Test BigQuery insertion error handling"""
    table = 'test-project.test_dataset.test_table'
    row = {'field1': 'value1'}

    mock_bigquery_client.insert_rows_json.return_value = ['error']

    with pytest.raises(SystemError, match='Encountered errors while inserting rows'):
        insert_to_biqquery(table, row)


def test_insert_to_bigquery_disabled(mock_bigquery_client, mock_config):
    """Test BigQuery insertion when disabled"""
    mock_config.send_to_bq = False

    insert_to_biqquery('table', {'field': 'value'})

    mock_bigquery_client.insert_rows_json.assert_not_called()


# Test Pub/Sub Operations
def test_publish_to_pubsub(mock_pubsub_client, mock_config):
    """Test successful Pub/Sub message publishing"""
    job_id = 'test-job'
    user_id = 'test-user'
    topic = 'test-topic'
    message = {'data': 'test'}

    mock_pubsub_client.topic_path.return_value = f'projects/{mock_config.gcp_project}/topics/{topic}'

    publish_to_pubsub(job_id, user_id, topic, message)

    mock_pubsub_client.publish.assert_called_once()
    # Verify the message contains required fields
    published_msg = mock_pubsub_client.publish.call_args[0][1].decode('utf-8')
    assert job_id in published_msg
    assert user_id in published_msg
    assert 'test' in published_msg


def test_publish_to_pubsub_disabled(mock_pubsub_client, mock_config):
    """Test Pub/Sub publishing when disabled"""
    mock_config.send_to_pubsub = False

    publish_to_pubsub('job-id', 'user-id', 'topic', {'data': 'test'})

    mock_pubsub_client.publish.assert_not_called()


# Test Storage Operations
def test_get_results_bucket_local(mock_config):
    """Test getting results bucket name for local environment"""
    mock_config.env_name = 'local'

    result = get_results_bucket()

    assert result == 'lum-pipeline-zen-jobs-local'


def test_get_results_bucket_with_suffix(mock_config):
    """Test getting results bucket name with custom suffix"""
    mock_config.results_bucket_suffix = 'custom'

    result = get_results_bucket()

    assert result == 'lum-pipeline-zen-jobs-custom'


@patch('common.gcp.get_zone_from_metadata')
def test_get_results_bucket_production(mock_get_zone):
    """Test getting results bucket name in production environment"""
    mock_get_zone.return_value = 'us-central1-a'

    with patch('common.gcp.is_local_env') as mock_is_local_env:
        mock_is_local_env.return_value = False
        result = get_results_bucket()

    assert result == 'lum-pipeline-zen-jobs-us'


def test_upload_directory(mock_storage_client, mock_config):
    """Test directory upload to GCS"""
    with patch('glob.glob') as mock_glob:
        mock_glob.return_value = ['/path/file1.txt', '/path/file2.txt']
        with patch('os.path.isfile') as mock_isfile, \
                patch('os.path.isdir') as mock_isdir:
            mock_isfile.return_value = True
            mock_isdir.return_value = True

            upload_directory('/path', 'test-bucket')

            mock_storage_client.get_bucket.assert_called_once_with('test-bucket')
            bucket = mock_storage_client.get_bucket.return_value
            assert bucket.blob.call_count == 2


def test_upload_file(mock_storage_client, mock_config):
    """Test single file upload to GCS"""
    local_file = '/path/to/file.txt'
    bucket_name = 'test-bucket'
    gcs_path = 'test/path'

    with patch('os.path.isfile') as mock_isfile:
        mock_isfile.return_value = True

        upload_file(local_file, bucket_name, gcs_path)

        mock_storage_client.get_bucket.assert_called_once_with(bucket_name)
        bucket = mock_storage_client.get_bucket.return_value
        bucket.blob.assert_called_once()
        blob = bucket.blob.return_value
        blob.upload_from_filename.assert_called_once_with(local_file)


@patch('common.gcp.upload_file')
@patch('common.gcp.get_results_bucket')
def test_upload_jobs_meta_success(mock_get_results_bucket, mock_upload_file, mock_config):
    """Test successful jobs metadata upload"""
    job_id = 'test-job'
    user_id = 'test-user'
    work_dir = '/test/work/dir'
    mock_get_results_bucket.return_value = 'pipeline-zen-jobs-us'

    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True

        upload_jobs_meta(job_id, user_id)

        mock_upload_file.assert_called_once()


@patch('common.gcp.get_work_dir')
def test_upload_jobs_meta_file_not_exists(mock_get_work_dir, mock_config):
    """Test jobs metadata upload when file doesn't exist"""
    mock_get_work_dir.return_value = '/test/work/dir'

    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = False

        upload_jobs_meta('test-job', 'test-user')
        # Should not raise an error, just log warning


def test_download_object(mock_storage_client):
    """Test downloading an object from GCS"""
    bucket_name = 'test-bucket'
    source_blob = 'source/path.txt'
    dest_file = 'local/path.txt'

    download_object(bucket_name, source_blob, dest_file)

    mock_storage_client.bucket.assert_called_once_with(bucket_name)
    bucket = mock_storage_client.bucket.return_value
    bucket.blob.assert_called_once_with(source_blob)
    blob = bucket.blob.return_value
    blob.download_to_filename.assert_called_once_with(dest_file)


# Test Metadata Operations
def test_get_vm_name_from_metadata(mock_requests, mock_config):
    """Test getting VM name from metadata"""
    expected_name = 'test-vm'
    mock_requests.get.return_value.text = expected_name

    result = get_vm_name_from_metadata()

    assert result == expected_name
    mock_requests.get.assert_called_once()


def test_get_zone_from_metadata(mock_requests, mock_config):
    """Test getting zone from metadata"""
    mock_requests.get.return_value.text = 'projects/123/zones/us-central1-a'

    result = get_zone_from_metadata()

    assert result == 'us-central1-a'
    mock_requests.get.assert_called_once()


# Test Helper Functions
def test_get_mig_name_from_vm_name():
    """Test extracting MIG name from VM name"""
    vm_name = 'pipeline-zen-jobs-8xa100-40gb-us-central1-asj3'

    result = get_mig_name_from_vm_name(vm_name)

    assert result == 'pipeline-zen-jobs-8xa100-40gb-us-central1'


def test_get_region_from_zone():
    """Test extracting region from zone"""
    zone = 'us-central1-a'

    result = get_region_from_zone(zone)

    assert result == 'us-central1'


def test_get_multi_region_from_zone():
    """Test extracting multi-region from zone"""
    zone = 'us-central1-a'

    result = get_multi_region_from_zone(zone)

    assert result == 'us'


def test_get_region_from_vm_name():
    """Test extracting region from VM name"""
    vm_name = 'pipeline-zen-jobs-8xa100-40gb-us-central1-asj3'

    result = get_region_from_vm_name(vm_name)

    assert result == 'us-central1'


# Error Cases
def test_upload_jobs_meta_too_many_requests(mock_storage_client, mock_config):
    """Test handling of TooManyRequests error during metadata upload"""
    with patch('common.gcp.upload_file') as mock_upload:
        mock_upload.side_effect = TooManyRequests("Too many requests")
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True

            # Should not raise an error, just log warning
            upload_jobs_meta('test-job', 'test-user')


def test_metadata_operations_not_gcp(mock_config):
    """Test metadata operations when not running on GCP"""
    mock_config.is_gcp = False

    assert get_vm_name_from_metadata() is None
    assert get_zone_from_metadata() is None


def test_storage_operations_disabled(mock_storage_client, mock_config):
    """Test storage operations when GCS is disabled"""
    mock_config.send_to_gcs = False

    upload_directory('/path', 'bucket')
    upload_file('/path/file.txt', 'bucket')
    make_object_public('bucket', 'object')

    mock_storage_client.get_bucket.assert_not_called()
