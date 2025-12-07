package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"cloud.google.com/go/cloudsqlconn"
	"cloud.google.com/go/cloudsqlconn/postgres/pgxv4"
)

// visitData is used to pass data to the HTML template.
type visitData struct {
  RecentVisits []visit
}

// visit contains a single row from the visits table in the database.
// Each visit includes a timestamp.
type visit struct {
  VisitTime time.Time
}

// getDB creates a connection to the database
// based on environment variables.
func getDB() (*sql.DB, func() error) {
  cleanup, err := pgxv4.RegisterDriver("cloudsql-postgres", cloudsqlconn.WithIAMAuthN())
  if err != nil {
    log.Fatalf("Error on pgxv4.RegisterDriver: %v", err)
  }

  dsn := fmt.Sprintf("host=%s user=%s dbname=%s sslmode=disable", os.Getenv("INSTANCE_CONNECTION_NAME"), os.Getenv("DB_USER"), os.Getenv("DB_NAME"))
  db, err := sql.Open("cloudsql-postgres", dsn)
  if err != nil {
    log.Fatalf("Error on sql.Open: %v", err)
  }

  createVisits := `CREATE TABLE IF NOT EXISTS visits (
    id SERIAL NOT NULL,
    created_at timestamp NOT NULL,
    PRIMARY KEY (id)
  );`
  _, err = db.Exec(createVisits)
  if err != nil {
    log.Fatalf("unable to create table: %s", err)
  }

  return db, cleanup
}

func main() {
  port := os.Getenv("PORT")
  if port == "" {
    port = "8080"
  }
  log.Printf("Listening on port %s", port)
  db, cleanup := getDB()
  defer cleanup()

  http.HandleFunc("/", func(w http.ResponseWriter, _ *http.Request) {
    // Insert current visit
    _, err := db.Exec("INSERT INTO visits(created_at) VALUES(NOW())")
    if err != nil {
      log.Fatalf("unable to save visit: %v", err)
    }

    // Get the last 5 visits
    rows, err := db.Query("SELECT created_at FROM visits ORDER BY created_at DESC LIMIT 5")
    if err != nil {
      log.Fatalf("DB.Query: %v", err)
    }
    defer rows.Close()

    var visits []visit
    for rows.Next() {
      var visitTime time.Time
      err := rows.Scan(&visitTime)
      if err != nil {
        log.Fatalf("Rows.Scan: %v", err)
      }
      visits = append(visits, visit{VisitTime: visitTime})
    }
    response, err := json.Marshal(visitData{RecentVisits: visits})
    if err != nil {
      log.Fatalf("renderIndex: failed to parse totals with json.Marshal: %v", err)
    }
    w.Write(response)
  })
  if err := http.ListenAndServe(":"+port, nil); err != nil {
    log.Fatal(err)
  }
}


