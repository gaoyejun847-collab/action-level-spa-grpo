"""Strict synchronous client used once per complete task batch on the trainer driver."""
import json
import math
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class PRMServiceError(RuntimeError):
    pass


class PRMClient:
    def __init__(self, url, expected_version, timeout=1800):
        if not expected_version:
            raise ValueError('An explicit frozen PRM version is required')
        self.url, self.expected_version, self.timeout = url.rstrip('/'), expected_version, timeout

    def _request(self, path, payload=None):
        request = Request(self.url + path,
                          data=None if payload is None else json.dumps(payload, allow_nan=False).encode(),
                          headers={'Content-Type': 'application/json'})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            raise PRMServiceError('Frozen PRM service unavailable; policy update must stop: ' + str(exc)) from exc
        if result.get('model_version') != self.expected_version:
            raise PRMServiceError('Frozen PRM model version changed; policy update must stop')
        return result

    def health(self):
        result = self._request('/health')
        if not (result.get('precision_gate_passed') is True or result.get('quantization_gate_passed') is True) or result.get('frozen') is not True:
            raise PRMServiceError('PRM service has not passed frozen precision qualification')
        return result

    def score(self, trajectories):
        for row in trajectories:
            if set(row) != {'input_ids', 'action_end_positions'}:
                raise ValueError('Never send reward labels or future-feedback fields to PRM')
        result = self._request('/score', dict(model_version=self.expected_version, trajectories=trajectories))
        rows = result.get('results')
        if not isinstance(rows, list) or len(rows) != len(trajectories):
            raise PRMServiceError('PRM returned an invalid batch shape')
        for item, trajectory in zip(rows, trajectories):
            scores = item.get('prm_scores', [])
            if (item.get('valid') is not True or not isinstance(scores, list) or
                    len(scores) != len(trajectory['action_end_positions']) or
                    any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in scores)):
                item.update(valid=False, prm_scores=[], error=item.get('error') or 'missing_or_nonfinite_prm_score')
        return rows
