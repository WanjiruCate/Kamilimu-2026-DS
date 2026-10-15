package kamilimu.powersource;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

/**
 * Android and plain JVM client for the power-source service.
 *
 * The web console at GET / posts the same JSON. A mobile app does not embed
 * the model; it sends the last few sensor readings and reads the label back.
 *
 *   PowerSourceClient client = new PowerSourceClient("http://10.0.2.2:8010");
 *   String json = client.predict(payload);
 */
public final class PowerSourceClient {
    private final HttpClient http;
    private final URI predictUrl;

    public PowerSourceClient(String baseUrl) {
        this.http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
        this.predictUrl = URI.create(baseUrl.replaceAll("/$", "") + "/predict");
    }

    public String predict(String jsonBody) throws IOException, InterruptedException {
        HttpRequest request = HttpRequest.newBuilder(predictUrl)
            .timeout(Duration.ofSeconds(10))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
            .build();
        HttpResponse<String> response = http.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() != 200) {
            throw new IOException("Power-source API returned " + response.statusCode() + ": " + response.body());
        }
        return response.body();
    }
}
