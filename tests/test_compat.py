from neural_ode._compat import to_plain_dict

def test_to_plain_dict():
    input_value = {"key1": "value1", "key2": {"key3": "value3"}}
     
    

    result = to_plain_dict(input_value)

    assert type(result) is dict
    assert type(result["key2"]) is dict
    assert result["key1"] == "value1"
    assert result["key2"] == {"key3": "value3"}
    assert result["key2"]["key3"] == "value3"

def test_to_plain_dict_one_value():
    input_value = 5

    result = to_plain_dict(input_value)

    assert type(result) is int
    assert result == 5