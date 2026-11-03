{# Keep the former fully qualified heading target for incoming links. #}
.. _{{ fullname | lower | replace('.', '-') | replace('_', '-') }}:

{% set title = objname %}
{% if fullname == 'pyrite.detectors.timepix_response.get_response' %}
{% set title = 'get_response (Timepix)' %}
{% elif fullname == 'pyrite.detectors.eaglexo_response.get_response' %}
{% set title = 'get_response (Eagle XO)' %}
{% endif %}
{{ title | escape | underline }}

Canonical import: ``{{ fullname }}``.
