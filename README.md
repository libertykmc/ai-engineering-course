# ai-engineering-course

## Troubleshooting

Ошибка 'ascii' codec can't encode characters

Что видел: при запуске программы с неправильным значением GIGACHAT_CREDENTIALS появилась ошибка: 'ascii' codec can't encode characters in position 7-15: ordinal not in range(128).

Почему: в GIGACHATCREDENTIALS был указан сломанный ключ с русскими символами. Из-за этого его не получилось корректно передать в HTTP-заголовке авторизации.

Что сделал: добавил обработку ошибки через try/except. Если запрос к GigaChat завершается ошибкой, программа автоматически переключается на HuggingFace и получает ответ от него.

Неверные данные авторизации GigaChat

Что видел: при запуске получил сообщение: Invalid credentials format. Please use only base64 credentials.

Почему: в GIGACHAT_CREDENTIALS был специально указан неправильный ключ вместо настоящих данных авторизации.

Что сделал: после проверки работы fallback вернул правильный GIGACHAT_CREDENTIALS. После этого запрос снова начал выполняться через GigaChat
